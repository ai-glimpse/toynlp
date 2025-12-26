"""ARC GRPO (Group Relative Policy Optimization) training for GPT model."""

import copy
import pathlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
import wandb
from datasets import Dataset, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from tokenizers import Tokenizer

from toynlp.gpt.config import GPTConfig
from toynlp.gpt.model import GPTModel
from toynlp.gpt.tokenizer import GPTTokenizer
from toynlp.paths import GPT_SFT_MODEL_PATH, GPT_ARC_RL_MODEL_PATH
from toynlp.util import current_device


EOS_TOKEN = "<eos>"


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""

    # Generation
    num_generations: int = 4  # Number of completions per prompt (G in GRPO)
    max_new_tokens: int = 8  # Max tokens to generate for answer
    temperature: float = 0.7  # Sampling temperature

    # Training
    epochs: int = 5
    batch_size: int = 8  # Number of prompts per batch
    learning_rate: float = 1e-5
    kl_coef: float = 0.1  # KL penalty coefficient (beta)
    clip_range: float = 0.2  # PPO-style clipping (optional)
    max_grad_norm: float = 1.0

    # Reward
    correct_reward: float = 1.0
    incorrect_reward: float = 0.0

    # Logging
    wandb_enabled: bool = True
    wandb_project: str = "toynlp"
    wandb_name: str = "gpt_arc_grpo"


def _format_arc_prompt(row: dict) -> dict:
    """Format ARC row as prompt (without answer)."""
    question_text = str(row.get("question") or "").strip()
    choices = row.get("choices") or {}
    labels = choices.get("label") or []
    texts = choices.get("text") or []
    choice_lines = [f"{label}. {text}" for label, text in zip(labels, texts, strict=False) if label and text]
    choice_block = "Choices:\n" + "\n".join(choice_lines) if choice_lines else ""

    instruction_parts = ["Answer the multiple-choice question with exactly one letter."]
    if question_text:
        instruction_parts.append(f"Question: {question_text}")
    if choice_block:
        instruction_parts.append(choice_block)
    instruction = "\n\n".join(instruction_parts)

    prompt = f"Human: {instruction}\n\nAssistant:"
    answer = str(row.get("answerKey") or "").strip().upper()
    valid_choices = [str(label).upper() for label in labels if label]

    return {"prompt": prompt, "answer": answer, "valid_choices": valid_choices}


def load_arc_prompts(split: str = "train") -> Dataset:
    """Load ARC dataset as prompts with answers."""
    configs = ["ARC-Challenge", "ARC-Easy"]
    datasets = []
    for config_name in configs:
        ds = load_dataset("allenai/ai2_arc", config_name, split=split, trust_remote_code=False)
        if isinstance(ds, Dataset):
            datasets.append(ds)
    combined = concatenate_datasets(datasets)
    return combined.map(_format_arc_prompt, remove_columns=combined.column_names, num_proc=4)


class GRPOTrainer:
    """GRPO trainer for ARC task."""

    def __init__(
        self,
        model: GPTModel,
        tokenizer: "Tokenizer",
        config: GRPOConfig,
        model_path: pathlib.Path = GPT_ARC_RL_MODEL_PATH,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model_path = model_path
        self.device = current_device

        # Create reference model (frozen copy for KL computation)
        self.ref_model = copy.deepcopy(model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=config.learning_rate,
        )

        # Token IDs
        self.eos_token_id = tokenizer.token_to_id("<eos>")
        self.pad_token_id = tokenizer.token_to_id("<pad>")

        # Running baseline for variance reduction
        self.reward_baseline = 0.0
        self.baseline_decay = 0.99

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode prompt to tensor."""
        token_ids = self.tokenizer.encode(prompt).ids
        return torch.tensor(token_ids, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def _generate_completions(
        self, prompt_ids: torch.Tensor, num_generations: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate multiple completions for a prompt.

        Returns:
            generated_ids: (num_generations, seq_len) - full sequences including prompt
            generated_mask: (num_generations, seq_len) - mask for generated tokens only
        """
        prompt_len = prompt_ids.size(0)
        batch_prompts = prompt_ids.unsqueeze(0).expand(num_generations, -1)

        # Initialize
        generated = batch_prompts.clone()
        finished = torch.zeros(num_generations, dtype=torch.bool, device=self.device)

        for _ in range(self.config.max_new_tokens):
            if finished.all():
                break

            logits = self.model(generated)
            next_token_logits = logits[:, -1, :] / self.config.temperature

            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)

            # Update finished status
            finished = finished | (next_tokens.squeeze(-1) == self.eos_token_id)

            # Append
            generated = torch.cat([generated, next_tokens], dim=1)

        # Create mask for generated tokens (excluding prompt)
        seq_len = generated.size(1)
        generated_mask = torch.zeros_like(generated, dtype=torch.bool)
        generated_mask[:, prompt_len:] = True

        return generated, generated_mask

    def _compute_log_probs(self, model: GPTModel, sequences: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities for generated tokens.

        Args:
            model: The model to compute log probs with
            sequences: (batch, seq_len) - full sequences
            mask: (batch, seq_len) - mask for tokens to compute log probs for

        Returns:
            log_probs: (batch,) - sum of log probs for masked tokens
        """
        logits = model(sequences[:, :-1])  # (batch, seq_len-1, vocab)
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather log probs for actual tokens
        target_tokens = sequences[:, 1:]  # (batch, seq_len-1)
        token_log_probs = log_probs.gather(2, target_tokens.unsqueeze(-1)).squeeze(-1)

        # Mask to only include generated tokens (shift mask by 1 for targets)
        target_mask = mask[:, 1:].float()

        # Sum log probs for each sequence
        return (token_log_probs * target_mask).sum(dim=-1)

    def _extract_answer(self, generated_text: str, prompt: str, valid_choices: list[str]) -> str:
        """Extract answer letter from generated text."""
        valid = [label.upper() for label in valid_choices if label]
        if not valid:
            return ""

        # Get only the completion part
        if generated_text.startswith(prompt):
            completion = generated_text[len(prompt) :].strip()
        else:
            completion = generated_text.strip()

        # Try to find a valid choice letter
        pattern = "|".join(re.escape(label) for label in sorted(valid, key=len, reverse=True))
        matches = list(re.finditer(rf"({pattern})", completion, re.IGNORECASE))
        return matches[0].group(1).upper() if matches else ""

    def _compute_rewards(
        self,
        generated_ids: torch.Tensor,
        prompt: str,
        answer: str,
        valid_choices: list[str],
    ) -> torch.Tensor:
        """Compute rewards for generated sequences.

        Returns:
            rewards: (num_generations,) - reward for each completion
        """
        num_gen = generated_ids.size(0)
        rewards = torch.zeros(num_gen, device=self.device)

        for i in range(num_gen):
            generated_text = self.tokenizer.decode(generated_ids[i].tolist())
            predicted = self._extract_answer(generated_text, prompt, valid_choices)

            if predicted == answer:
                rewards[i] = self.config.correct_reward
            else:
                rewards[i] = self.config.incorrect_reward

        return rewards

    def _compute_group_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute group-relative advantages (GRPO core idea).

        Normalize rewards within the group to get advantages.
        """
        mean_reward = rewards.mean()
        std_reward = rewards.std() + 1e-8

        # Group-relative normalization
        advantages = (rewards - mean_reward) / std_reward

        return advantages

    def train_step(self, batch: list[dict]) -> dict[str, float]:
        """Perform one GRPO training step on a batch of prompts.

        Args:
            batch: List of dicts with 'prompt', 'answer', 'valid_choices'

        Returns:
            metrics: Dict with loss, reward, accuracy, etc.
        """
        self.model.train()
        total_loss = 0.0
        total_reward = 0.0
        total_correct = 0
        total_samples = 0
        total_kl = 0.0

        for item in batch:
            prompt = item["prompt"]
            answer = item["answer"]
            valid_choices = item["valid_choices"]

            # Encode prompt
            prompt_ids = self._encode_prompt(prompt)

            # Generate G completions
            self.model.eval()
            with torch.no_grad():
                generated_ids, generated_mask = self._generate_completions(prompt_ids, self.config.num_generations)

            # Compute rewards
            rewards = self._compute_rewards(generated_ids, prompt, answer, valid_choices)

            # Skip if all rewards are the same (no learning signal)
            if rewards.std() < 1e-8:
                total_reward += rewards.mean().item()
                total_correct += (rewards == self.config.correct_reward).sum().item()
                total_samples += self.config.num_generations
                continue

            # Compute advantages (group-relative)
            advantages = self._compute_group_advantages(rewards)

            # Compute log probs under current policy
            self.model.train()
            current_log_probs = self._compute_log_probs(self.model, generated_ids, generated_mask)

            # Compute log probs under reference policy (for KL)
            with torch.no_grad():
                ref_log_probs = self._compute_log_probs(self.ref_model, generated_ids, generated_mask)

            # KL divergence (approximate)
            kl_div = (current_log_probs - ref_log_probs).mean()

            # Policy gradient loss with advantages
            # Loss = -E[advantage * log_prob] + beta * KL
            pg_loss = -(advantages * current_log_probs).mean()
            kl_loss = self.config.kl_coef * kl_div

            loss = pg_loss + kl_loss

            # Accumulate
            total_loss += loss.item()
            total_reward += rewards.mean().item()
            total_correct += (rewards == self.config.correct_reward).sum().item()
            total_samples += self.config.num_generations
            total_kl += kl_div.item()

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

        num_items = len(batch)
        return {
            "loss": total_loss / max(num_items, 1),
            "reward": total_reward / max(num_items, 1),
            "accuracy": total_correct / max(total_samples, 1),
            "kl": total_kl / max(num_items, 1),
        }

    def evaluate(self, split: str = "test", max_samples: int | None = None) -> dict[str, float]:
        """Evaluate model accuracy on dataset, returning per-config and overall accuracy."""
        self.model.eval()

        results = {}
        overall_correct = 0
        overall_total = 0

        for config_name in ["ARC-Challenge", "ARC-Easy"]:
            ds = load_dataset("allenai/ai2_arc", config_name, split=split, trust_remote_code=False)
            if not isinstance(ds, Dataset):
                continue
            ds = ds.map(_format_arc_prompt, remove_columns=ds.column_names, num_proc=4)

            if max_samples is not None:
                ds = ds.select(range(min(max_samples, len(ds))))

            correct = 0
            total = 0

            with torch.no_grad():
                for item in ds:
                    prompt = item["prompt"]
                    answer = item["answer"]
                    valid_choices = item["valid_choices"]

                    prompt_ids = self._encode_prompt(prompt)

                    # Greedy generation for eval
                    generated = prompt_ids.unsqueeze(0)
                    for _ in range(self.config.max_new_tokens):
                        logits = self.model(generated)
                        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                        generated = torch.cat([generated, next_token], dim=1)
                        if next_token.item() == self.eos_token_id:
                            break

                    generated_text = self.tokenizer.decode(generated[0].tolist())
                    predicted = self._extract_answer(generated_text, prompt, valid_choices)

                    if predicted == answer:
                        correct += 1
                    total += 1

            accuracy = correct / max(total, 1)
            results[f"{config_name}_accuracy"] = accuracy
            results[f"{config_name}_correct"] = correct
            results[f"{config_name}_total"] = total
            overall_correct += correct
            overall_total += total

        results["accuracy"] = overall_correct / max(overall_total, 1)
        results["correct"] = overall_correct
        results["total"] = overall_total

        return results

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        test_dataset: Dataset | None = None,
    ) -> None:
        """Run GRPO training loop."""
        best_accuracy = 0.0

        # Create dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=lambda x: x,  # Keep as list of dicts
        )

        for epoch in range(self.config.epochs):
            epoch_metrics = {"loss": 0.0, "reward": 0.0, "accuracy": 0.0, "kl": 0.0}
            num_batches = 0

            for batch in train_loader:
                step_metrics = self.train_step(batch)

                for key in epoch_metrics:
                    epoch_metrics[key] += step_metrics[key]
                num_batches += 1

                # Log batch metrics
                if self.config.wandb_enabled:
                    wandb.log(
                        {
                            "grpo/batch_loss": step_metrics["loss"],
                            "grpo/batch_reward": step_metrics["reward"],
                            "grpo/batch_accuracy": step_metrics["accuracy"],
                            "grpo/batch_kl": step_metrics["kl"],
                        }
                    )

            # Average epoch metrics
            for key in epoch_metrics:
                epoch_metrics[key] /= max(num_batches, 1)

            # Evaluate
            train_eval_metrics = {"accuracy": 0.0, "ARC-Challenge_accuracy": 0.0, "ARC-Easy_accuracy": 0.0}
            val_metrics = {"accuracy": 0.0, "ARC-Challenge_accuracy": 0.0, "ARC-Easy_accuracy": 0.0}
            test_metrics = {"accuracy": 0.0, "ARC-Challenge_accuracy": 0.0, "ARC-Easy_accuracy": 0.0}

            train_eval_metrics = self.evaluate(split="train")
            if val_dataset is not None:
                val_metrics = self.evaluate(split="validation")
            if test_dataset is not None:
                test_metrics = self.evaluate(split="test")

            print(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"Loss: {epoch_metrics['loss']:.4f}, "
                f"KL: {epoch_metrics['kl']:.4f}"
            )
            print(
                f"  Train - Challenge: {train_eval_metrics['ARC-Challenge_accuracy']:.2%}, "
                f"Easy: {train_eval_metrics['ARC-Easy_accuracy']:.2%}, "
                f"Overall: {train_eval_metrics['accuracy']:.2%}"
            )
            print(
                f"  Val - Challenge: {val_metrics['ARC-Challenge_accuracy']:.2%}, "
                f"Easy: {val_metrics['ARC-Easy_accuracy']:.2%}, "
                f"Overall: {val_metrics['accuracy']:.2%}"
            )
            print(
                f"  Test - Challenge: {test_metrics['ARC-Challenge_accuracy']:.2%}, "
                f"Easy: {test_metrics['ARC-Easy_accuracy']:.2%}, "
                f"Overall: {test_metrics['accuracy']:.2%}"
            )

            # Save best model
            if val_metrics["accuracy"] > best_accuracy:
                best_accuracy = val_metrics["accuracy"]
                self.save_model()
                print(f"Saved best model (accuracy: {best_accuracy:.2%})")

            # Log epoch metrics
            if self.config.wandb_enabled:
                wandb.log(
                    {
                        "grpo/epoch": epoch + 1,
                        "grpo/train_loss": epoch_metrics["loss"],
                        "grpo/train_reward": epoch_metrics["reward"],
                        "grpo/kl": epoch_metrics["kl"],
                        "grpo/train_accuracy": train_eval_metrics["accuracy"],
                        "grpo/train_challenge_accuracy": train_eval_metrics["ARC-Challenge_accuracy"],
                        "grpo/train_easy_accuracy": train_eval_metrics["ARC-Easy_accuracy"],
                        "grpo/val_accuracy": val_metrics["accuracy"],
                        "grpo/val_challenge_accuracy": val_metrics["ARC-Challenge_accuracy"],
                        "grpo/val_easy_accuracy": val_metrics["ARC-Easy_accuracy"],
                        "grpo/test_accuracy": test_metrics["accuracy"],
                        "grpo/test_challenge_accuracy": test_metrics["ARC-Challenge_accuracy"],
                        "grpo/test_easy_accuracy": test_metrics["ARC-Easy_accuracy"],
                    }
                )

    def save_model(self) -> None:
        """Save model state dict."""
        torch.save(self.model.state_dict(), self.model_path)


def train_arc_grpo(
    pretrained_model_path: pathlib.Path = GPT_SFT_MODEL_PATH,
    save_path: pathlib.Path = GPT_ARC_RL_MODEL_PATH,
    grpo_config: GRPOConfig | None = None,
) -> None:
    """Train ARC with GRPO starting from SFT model.

    Args:
        pretrained_model_path: Path to pretrained model (default: general SFT model)
        save_path: Path to save GRPO-trained model
        grpo_config: GRPO configuration
    """
    if grpo_config is None:
        grpo_config = GRPOConfig()

    if grpo_config.wandb_enabled:
        wandb.init(
            project=grpo_config.wandb_project,
            name=grpo_config.wandb_name,
            config=vars(grpo_config),
        )

    print("\n" + "=" * 60)
    print("ARC GRPO Training")
    print("=" * 60)

    # Load tokenizer
    tokenizer = GPTTokenizer().load()
    pad_token_id = tokenizer.token_to_id("<pad>")

    # Load model
    loaded = torch.load(pretrained_model_path, map_location=current_device, weights_only=False)
    if isinstance(loaded, dict):
        model = GPTModel(GPTConfig(), padding_idx=pad_token_id)
        model.load_state_dict(loaded)
    else:
        model = loaded
    model.to(current_device)

    # Load datasets
    print("Loading ARC datasets...")
    train_dataset = load_arc_prompts("train")
    val_dataset = load_arc_prompts("validation")
    test_dataset = load_arc_prompts("test")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=grpo_config,
        model_path=save_path,
    )

    # Train
    trainer.train(train_dataset, val_dataset, test_dataset)

    print("\n" + "=" * 60)
    print("GRPO training completed!")
    print(f"Model saved to: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    config = GRPOConfig(
        num_generations=4,
        epochs=5,
        batch_size=8,
        learning_rate=1e-5,
        kl_coef=0.1,
        temperature=0.7,
        wandb_enabled=True,
        wandb_name="gpt_arc_grpo",
    )
    train_arc_grpo(grpo_config=config)
