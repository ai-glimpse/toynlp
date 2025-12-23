"""ARC-specific SFT (Supervised Fine-Tuning) for GPT model."""

import pathlib
import random
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import wandb
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from tokenizers import Tokenizer

from toynlp.gpt.config import GPTConfig
from toynlp.gpt.model import GPTModel
from toynlp.gpt.sft import (
    GPTSFTTrainer,
    get_sft_dataloaders,
    apply_lora,
    mark_only_lora_as_trainable,
)
from toynlp.gpt.tokenizer import GPTTokenizer
from toynlp.paths import GPT_SFT_MODEL_PATH, GPT_ARC_MODEL_PATH
from toynlp.util import current_device


@dataclass
class ARCEvalResult:
    accuracy: float
    total: int
    correct: int


class ARCEvaluator:
    """Lightweight ARC evaluator for use during training."""

    def __init__(self, model: GPTModel, tokenizer: "Tokenizer", config: GPTConfig) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = current_device
        self._eval_datasets: dict[str, Dataset] | None = None

    def _load_eval_datasets(self) -> dict[str, Dataset]:
        """Lazy load evaluation datasets."""
        if self._eval_datasets is None:
            self._eval_datasets = {}
            for arc_config in ("ARC-Challenge", "ARC-Easy"):
                dataset = load_dataset("allenai/ai2_arc", arc_config, split="test", trust_remote_code=False)
                if isinstance(dataset, Dataset):
                    self._eval_datasets[arc_config] = dataset
        return self._eval_datasets

    def _format_prompt(self, row: dict) -> str:
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
        return f"Human: {instruction}\n\nAssistant:"

    def _generate(self, prompt: str, max_length: int = 16) -> str:
        """Generate text from prompt."""
        input_ids = self.tokenizer.encode(prompt).ids
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)

        with torch.no_grad():
            for _ in range(max_length):
                outputs = self.model(input_tensor)
                next_token_logits = outputs[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                input_tensor = torch.cat([input_tensor, next_token_id], dim=1)
                if next_token_id.item() == self.tokenizer.token_to_id("<eos>"):
                    break

        return self.tokenizer.decode(input_tensor[0].tolist())

    def _extract_choice(self, generated: str, prompt: str, valid_labels: list[str]) -> str:
        valid = [label.upper() for label in valid_labels if label]
        if not valid:
            return ""
        pattern = "|".join(re.escape(label) for label in sorted(valid, key=len, reverse=True))
        completion = generated[len(prompt) :].lstrip() if generated.startswith(prompt) else generated.strip()
        matches = list(re.finditer(rf"({pattern})", completion, re.IGNORECASE))
        return matches[-1].group(1).upper() if matches else ""

    def evaluate(self, max_samples: int | None = None) -> dict[str, ARCEvalResult]:
        """Run ARC evaluation and return results per split."""
        self.model.eval()
        datasets = self._load_eval_datasets()
        results: dict[str, ARCEvalResult] = {}
        overall_correct = 0
        overall_total = 0

        for config_name, dataset in datasets.items():
            ds = dataset
            if max_samples is not None:
                ds = ds.select(range(min(max_samples, len(ds))))

            correct = 0
            total = len(ds)

            for row in ds:
                prompt = self._format_prompt(row)
                choices = row.get("choices") or {}
                labels = choices.get("label") or []
                generated = self._generate(prompt)
                predicted = self._extract_choice(generated, prompt, list(labels))
                answer = str(row.get("answerKey") or "").strip().upper()
                if predicted == answer:
                    correct += 1

            accuracy = correct / total if total > 0 else 0.0
            results[config_name] = ARCEvalResult(accuracy=accuracy, total=total, correct=correct)
            overall_correct += correct
            overall_total += total

        overall_acc = overall_correct / overall_total if overall_total > 0 else 0.0
        results["Overall"] = ARCEvalResult(accuracy=overall_acc, total=overall_total, correct=overall_correct)
        return results


class ARCTrainer(GPTSFTTrainer):
    """ARC trainer with per-epoch evaluation on ARC test datasets."""

    def __init__(
        self,
        config: GPTConfig,
        pad_token_id: int,
        model: GPTModel,
        model_path: pathlib.Path,
        tokenizer: "Tokenizer",
        eval_max_samples: int | None = None,
    ) -> None:
        super().__init__(config, pad_token_id, model, model_path, metric_prefix="arc")
        self.tokenizer = tokenizer
        self.eval_max_samples = eval_max_samples
        self._evaluator: ARCEvaluator | None = None

    @property
    def evaluator(self) -> ARCEvaluator:
        if self._evaluator is None:
            self._evaluator = ARCEvaluator(self.model, self.tokenizer, self.config)
        return self._evaluator

    def train(
        self,
        train_dataloader: DataLoader[dict[str, torch.Tensor]],
        val_dataloader: DataLoader[dict[str, torch.Tensor]],
        test_dataloader: DataLoader[dict[str, torch.Tensor]],
    ) -> None:
        """Train with per-batch loss logging and per-epoch ARC evaluation."""
        best_val_loss = float("inf")
        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss = 0.0
            total_samples = 0

            for batch in train_dataloader:
                self.update_lr()
                self.optimizer.zero_grad()

                batch_input_ids = batch["input_ids"].to(self.device)
                batch_target_ids = batch["labels"].to(self.device)

                input_ids = batch_input_ids[:, :-1]
                target_ids = batch_target_ids[:, 1:]

                logits = self.model(input_ids)
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

                if self.config.wandb_enabled:
                    wandb.log({f"{self.metric_prefix}/train_batch_loss": loss.item()})

                if random.random() < 0.01:
                    self._print_sample_predictions(input_ids[0], target_ids[0], logits[0], "train")

                loss.backward()
                if self.clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                self.optimizer.step()

                self.current_step += 1
                batch_size = input_ids.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

            avg_train_loss = total_loss / total_samples
            train_perplexity = torch.exp(torch.tensor(avg_train_loss)).item()

            # Validate (epoch-level only)
            self.model.eval()
            with torch.no_grad():
                val_loss_stats = self.calc_loss_loader(val_dataloader, "val")
                test_loss_stats = self.calc_loss_loader(test_dataloader, "test")

            # Run ARC evaluation
            arc_results = self.evaluator.evaluate(max_samples=self.eval_max_samples)

            current_lr = self.get_lr()

            # Print results
            arc_overall = arc_results.get("Overall", ARCEvalResult(0, 0, 0))
            arc_challenge = arc_results.get("ARC-Challenge", ARCEvalResult(0, 0, 0))
            arc_easy = arc_results.get("ARC-Easy", ARCEvalResult(0, 0, 0))

            print(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, Train PPL: {train_perplexity:.4f}, "
                f"Val Loss: {val_loss_stats['loss']:.4f}, "
                f"Test Loss: {test_loss_stats['loss']:.4f}, "
                f"LR: {current_lr:.6f}"
            )
            print(
                f"  ARC Eval - Challenge: {arc_challenge.accuracy:.2%} ({arc_challenge.correct}/{arc_challenge.total})"
            )
            print(
                f"  ARC Eval - Easy: {arc_easy.accuracy:.2%} ({arc_easy.correct}/{arc_easy.total}), "
                f"Overall: {arc_overall.accuracy:.2%} ({arc_overall.correct}/{arc_overall.total})"
            )

            if val_loss_stats["loss"] < best_val_loss:
                best_val_loss = val_loss_stats["loss"]
                self.save_model()
                print(f"Saved best model at epoch {epoch + 1}")

            if self.config.wandb_enabled:
                wandb.log(
                    {
                        f"{self.metric_prefix}/epoch": epoch + 1,
                        f"{self.metric_prefix}/train_loss": avg_train_loss,
                        f"{self.metric_prefix}/train_perplexity": train_perplexity,
                        f"{self.metric_prefix}/val_loss": val_loss_stats["loss"],
                        f"{self.metric_prefix}/val_perplexity": val_loss_stats["perplexity"],
                        f"{self.metric_prefix}/test_loss": test_loss_stats["loss"],
                        f"{self.metric_prefix}/test_perplexity": test_loss_stats["perplexity"],
                        f"{self.metric_prefix}/learning_rate": current_lr,
                        f"{self.metric_prefix}/arc_challenge_acc": arc_challenge.accuracy,
                        f"{self.metric_prefix}/arc_easy_acc": arc_easy.accuracy,
                        f"{self.metric_prefix}/arc_overall_acc": arc_overall.accuracy,
                    }
                )


def train_arc_sft(
    pretrained_model_path: pathlib.Path = GPT_SFT_MODEL_PATH,
    save_path: pathlib.Path = GPT_ARC_MODEL_PATH,
    config: GPTConfig | None = None,
    wandb_enabled: bool = True,
) -> None:
    """Train ARC-specific SFT on a pretrained GPT model.

    Args:
        pretrained_model_path: Path to the pretrained model (default: general SFT model)
        save_path: Path to save the fine-tuned model (default: ARC model path)
        config: GPTConfig for training (uses defaults if None)
        wandb_enabled: Whether to enable wandb logging
    """
    if config is None:
        config = GPTConfig(
            wandb_enabled=wandb_enabled,
            wandb_name="gpt_arc_sft",
        )

    if config.wandb_enabled and wandb_enabled:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_name,
            config=config.to_dict(),
        )

    print("\n" + "=" * 60)
    print("ARC SFT Training")
    print("=" * 60)

    # Load tokenizer and model
    tokenizer = GPTTokenizer().load()
    padding_token_id = tokenizer.token_to_id("<pad>")

    # Load the model - handle both model objects and state dicts
    loaded = torch.load(pretrained_model_path, map_location=current_device, weights_only=False)

    if isinstance(loaded, dict):
        # If it's a state dict, create a model and load the weights
        model = GPTModel(config, padding_idx=padding_token_id)
        model.load_state_dict(loaded)
    else:
        # If it's already a model object
        model = loaded

    model.to(current_device)

    # Apply LoRA with higher dropout to reduce overfitting
    model = apply_lora(model, r=8, alpha=16, dropout=0.2)
    mark_only_lora_as_trainable(model)

    # Load ARC dataset
    train_dataloader, val_dataloader, test_dataloader = get_sft_dataloaders(
        config=config,
        gpt_tokenizer=tokenizer,
        dataset_names=["allenai/ai2_arc"],
    )

    # Create ARC trainer with per-epoch evaluation
    trainer = ARCTrainer(
        config=config,
        pad_token_id=padding_token_id,
        model=model,
        model_path=save_path,
        tokenizer=tokenizer,
        eval_max_samples=None,  # Set to a number to limit eval samples per epoch
    )
    # Lower learning rate and fewer epochs to reduce overfitting
    trainer.base_lr = 1e-5
    trainer.config.epochs = 5
    trainer.train(train_dataloader, val_dataloader, test_dataloader)

    print("\n" + "=" * 60)
    print("ARC SFT training completed!")
    print(f"Model saved to: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    config = GPTConfig(
        wandb_enabled=True,
        wandb_name="gpt_arc_sft",
    )
    train_arc_sft(config=config)
