import random
import torch
from torch.utils.data import DataLoader

import wandb
from toynlp.util import current_device
from toynlp.paths import GPT_MODEL_PATH
from toynlp.gpt.config import GPTConfig, create_config_from_cli
from toynlp.gpt.model import GPTModel
from toynlp.gpt.tokenizer import GPTTokenizer
from toynlp.util import setup_seed, set_deterministic_mode
from toynlp.gpt.dataset import get_split_dataloader


setup_seed(1234)  # Set a random seed for reproducibility
set_deterministic_mode()  # Set deterministic mode for reproducibility


class GPTTrainer:
    def __init__(self, config: GPTConfig, pad_token_id: int) -> None:
        self.config = config
        self.model = GPTModel(self.config, pad_token_id)
        self.model_path = GPT_MODEL_PATH
        self.device = current_device
        self.model.to(self.device)
        self.tokenizer = GPTTokenizer().load()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction="mean")
        self.clip_norm = self.config.clip_norm
        if self.clip_norm:
            print(f"Gradient clipping enabled with norm {self.clip_norm}")

        # Learning rate scheduler setup
        self.warmup_steps = self.config.warmup_steps  # As per GPT paper
        self.current_step = 0
        self.base_lr = self.config.learning_rate
        print(f"Learning rate warmup enabled: {self.warmup_steps} warmup steps")

    def get_lr(self) -> float:
        """Calculate learning rate with warmup and then keep constant."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * self.current_step / self.warmup_steps

        # Keep learning rate constant after warmup
        return self.base_lr

    def update_lr(self) -> None:
        """Update learning rate for all parameter groups."""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
    ) -> None:
        best_val_loss = float("inf")
        for epoch in range(self.config.epochs):
            train_stats = self._train_epoch(train_dataloader)
            val_loss_stats, test_loss_stats = self._validate_epoch(val_dataloader, test_dataloader)

            current_lr = self.get_lr()
            print(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"Train Loss: {train_stats['loss']:.4f}, "
                f"Train Perplexity: {train_stats['perplexity']:.4f}, "
                f"LR: {current_lr:.6f}, "
                f"Val Loss: {val_loss_stats['loss']:.4f}, "
                f"Val Perplexity: {val_loss_stats['perplexity']:.4f}, "
                f"Test Loss: {test_loss_stats['loss']:.4f}, "
                f"Test Perplexity: {test_loss_stats['perplexity']:.4f}"
            )
            if val_loss_stats["loss"] < best_val_loss:
                best_val_loss = val_loss_stats["loss"]
                torch.save(self.model, self.model_path)
                print(f"Saved best model({val_loss_stats['loss']:.4f}) from epoch {epoch + 1} to {self.model_path}")
            # log metrics to wandb
            if self.config.wandb_enabled:
                wandb.log(
                    {
                        "TrainLoss": train_stats["loss"],
                        "TrainPerplexity": train_stats["perplexity"],
                        "LearningRate": current_lr,
                        "Step": self.current_step,
                        "ValLoss": val_loss_stats["loss"],
                        "TestLoss": test_loss_stats["loss"],
                        "ValPerplexity": val_loss_stats["perplexity"],
                        "TestPerplexity": test_loss_stats["perplexity"],
                    },
                )

    def _train_epoch(self, train_dataloader: DataLoader) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in train_dataloader:
            # Update learning rate before each step
            self.update_lr()

            self.optimizer.zero_grad()
            batch_input_ids = batch["input_ids"].to(self.device)

            # For causal language modeling, input is shifted by 1 position
            # Input: [BOS, token1, token2, token3]
            # Target: [token1, token2, token3, EOS]
            input_ids = batch_input_ids[:, :-1]  # Remove last token
            target_ids = batch_input_ids[:, 1:]  # Remove first token (BOS)

            # Forward pass
            logits = self.model(input_ids)

            # Calculate loss
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

            # Print sample predictions occasionally
            if random.random() < 0.01:  # Print 1% of the batches
                self._print_sample_predictions(input_ids[0], target_ids[0], logits[0], "train")

            loss.backward()
            if self.clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
            self.optimizer.step()

            # Increment step counter
            self.current_step += 1

            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_train_loss = total_loss / total_samples
        perplexity = torch.exp(torch.tensor(avg_train_loss)).item()

        train_stats = {
            "loss": avg_train_loss,
            "perplexity": perplexity,
        }

        return train_stats

    def _print_sample_predictions(
        self, input_ids: torch.Tensor, target_ids: torch.Tensor, logits: torch.Tensor, data_split: str = "train"
    ) -> None:
        """Print sample input, target, and predicted tokens for debugging."""
        pred_ids = logits.argmax(dim=-1)

        print("=" * 100)
        print(f"[{data_split.upper()}] Sample Predictions:")
        print("Input tokens:", " | ".join([self.tokenizer.decode([int(token_id)]) for token_id in input_ids[:50]]))
        print("Target tokens:", " | ".join([self.tokenizer.decode([int(token_id)]) for token_id in target_ids[:50]]))
        print("Predicted tokens:", " | ".join([self.tokenizer.decode([int(token_id)]) for token_id in pred_ids[:50]]))
        print("=" * 100)

    def _validate_epoch(
        self,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
    ) -> tuple[dict[str, float], dict[str, float]]:
        self.model.eval()
        with torch.no_grad():
            val_loss_stats = self.calc_loss_loader(val_dataloader, "val")
            test_loss_stats = self.calc_loss_loader(test_dataloader, "test")
        return val_loss_stats, test_loss_stats

    def calc_loss_loader(self, data_loader: DataLoader, data_split: str = "val") -> dict[str, float]:
        total_loss = 0.0
        total_samples = 0

        for batch in data_loader:
            batch_input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            batch_size = batch_input_ids.size(0)

            # For causal language modeling, input is shifted by 1 position
            input_ids = batch_input_ids[:, :-1]  # Remove last token
            target_ids = batch_input_ids[:, 1:]  # Remove first token (BOS)

            # Forward pass
            logits = self.model(input_ids)

            # Calculate loss
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

            # Print sample predictions occasionally for validation/test
            if random.random() < 0.01:  # Print 1% of the batches
                self._print_sample_predictions(input_ids[0], target_ids[0], logits[0], data_split)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            torch.cuda.empty_cache()  # Clear cache to prevent OOM

        avg_loss = total_loss / total_samples
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        stats = {
            "loss": avg_loss,
            "perplexity": perplexity,
        }

        return stats


def train_model(config: GPTConfig) -> None:
    """Train the GPT model with the given configuration."""
    if config.wandb_enabled:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_name,
            config=config.to_dict(),
        )

    tokenizer = GPTTokenizer().load()

    train_dataloader = get_split_dataloader(
        config.dataset_path,
        config.dataset_split_of_model_train,
        config=config,
        gpt_tokenizer=tokenizer,
    )
    val_dataloader = get_split_dataloader(
        config.dataset_path,
        config.dataset_split_of_model_val,
        config=config,
        gpt_tokenizer=tokenizer,
    )
    test_dataloader = get_split_dataloader(
        config.dataset_path,
        config.dataset_split_of_model_test,
        config=config,
        gpt_tokenizer=tokenizer,
    )

    trainer = GPTTrainer(config=config, pad_token_id=tokenizer.token_to_id("<pad>"))
    trainer.train(train_dataloader, val_dataloader, test_dataloader)


def main() -> None:
    """CLI entry point for training GPT model using tyro configuration."""
    # Load configuration from command line using tyro
    config = create_config_from_cli()

    print("=" * 60)
    print("GPT MODEL TRAINING")
    print("=" * 60)
    print(f"Dataset: {config.dataset_path}")
    print(f"Training: {config.epochs} epochs, lr={config.learning_rate}")
    print(f"WandB: {'enabled' if config.wandb_enabled else 'disabled'}")
    print("=" * 60)

    train_model(config)


if __name__ == "__main__":
    main()
