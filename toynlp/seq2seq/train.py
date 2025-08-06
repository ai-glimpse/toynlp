import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
import argparse

import wandb
from toynlp.util import current_device
from toynlp.paths import SEQ2SEQ_MODEL_PATH
from toynlp.seq2seq.config import get_config, load_config
from toynlp.seq2seq.dataset import get_split_dataloader
from toynlp.seq2seq.model import Seq2SeqModel
from toynlp.seq2seq.tokenizer import Seq2SeqTokenizer
from toynlp.util import setup_seed, set_deterministic_mode

setup_seed(1234)  # Set a random seed for reproducibility
set_deterministic_mode()  # Set deterministic mode for reproducibility


class Seq2SeqTrainer:
    def __init__(self, pad_token_id: int) -> None:
        self.config = get_config()
        self.model = Seq2SeqModel(self.config.model)
        self.model_path = SEQ2SEQ_MODEL_PATH
        self.device = current_device
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.optimizer.learning_rate,
            weight_decay=self.config.optimizer.weight_decay,
        )
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
        self.clip_norm = self.config.training.clip_norm
        if self.clip_norm:
            print(f"Gradient clipping enabled with norm {self.clip_norm}")

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
    ) -> None:
        best_val_loss = float("inf")
        for epoch in range(self.config.training.epochs):
            train_loss = self._train_epoch(train_dataloader)
            val_loss, test_loss = self._validate_epoch(val_dataloader, test_dataloader)

            print(
                f"Epoch {epoch + 1}/{self.config.training.epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Test Loss: {test_loss:.4f}",
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model, self.model_path)
                print(f"Saved best model({val_loss=:.4f}) from epoch {epoch + 1} to {self.model_path}")

            # log metrics to wandb
            if self.config.wandb.enabled:
                wandb.log(
                    {
                        "TrainLoss": train_loss,
                        "ValLoss": val_loss,
                        "TestLoss": test_loss,
                        "TrainPerplexity": torch.exp(torch.tensor(train_loss)),
                        "ValPerplexity": torch.exp(torch.tensor(val_loss)),
                        "TestPerplexity": torch.exp(torch.tensor(test_loss)),
                    },
                )

    def _train_epoch(self, train_dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        for input_batch, target_batch in train_dataloader:
            self.optimizer.zero_grad()
            input_batch_device, target_batch_device = (
                input_batch.to(self.device),
                target_batch.to(self.device),
            )
            loss = self.calc_loss_batch(input_batch_device, target_batch_device)
            loss.backward()
            if self.clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
            self.optimizer.step()
            total_loss += loss.item() * input_batch.size(0)  # Multiply by batch size
            total_samples += input_batch.size(0)
        train_loss = total_loss / total_samples
        return train_loss

    def _validate_epoch(self, val_dataloader: DataLoader, test_dataloader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        with torch.no_grad():
            val_loss = self.calc_loss_loader(val_dataloader)
            test_loss = self.calc_loss_loader(test_dataloader)
        return val_loss, test_loss

    def calc_loss_batch(self, input_batch: torch.Tensor, target_batch: torch.Tensor) -> torch.Tensor:
        # Prepare logits and targets for loss calculation:
        # - We remove the first token in the sequence ([:, 1:, :]) because, in teacher forcing,
        #   the first output token is typically not used for loss (it corresponds to the start token).
        # - We then flatten the logits to shape (batch_size * seq_len_minus1, vocab_size)
        #   and the targets to shape (batch_size * seq_len_minus1), as required by nn.CrossEntropyLoss.
        #   This aligns each predicted token with its corresponding target token across the batch.
        logits = self.model(input_batch, target_batch)
        pred = logits[:, 1:, :].reshape(-1, logits.shape[-1])
        target_batch = target_batch[:, 1:].reshape(-1)
        loss = self.criterion(pred, target_batch)
        return loss

    def calc_loss_loader(self, data_loader: DataLoader) -> float:
        total_loss = 0.0
        total_samples = 0  # Track total samples
        for input_batch, target_batch in data_loader:
            input_batch_device, target_batch_device = (
                input_batch.to(self.device),
                target_batch.to(self.device),
            )
            loss = self.calc_loss_batch(input_batch_device, target_batch_device)
            total_loss += loss.item() * input_batch.size(0)  # Multiply by batch size
            total_samples += input_batch.size(0)
        return total_loss / total_samples  # Correct average


def train_model() -> None:
    config = get_config()
    if config.wandb.enabled:
        wandb.init(
            project=config.wandb.project,
            name=config.wandb.name,
            config=config.to_dict(),
        )

    dataset = load_dataset(path=config.dataset.path, name=config.dataset.name)
    source_tokenizer = Seq2SeqTokenizer(lang=config.dataset.source).load()
    target_tokenizer = Seq2SeqTokenizer(lang=config.dataset.target).load()

    train_dataloader = get_split_dataloader(
        dataset,  # type: ignore[unknown-argument]
        "train",
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        data_config=config.data,
    )
    val_dataloader = get_split_dataloader(
        dataset,  # type: ignore[unknown-argument]
        "validation",
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        data_config=config.data,
    )
    test_dataloader = get_split_dataloader(
        dataset,  # type: ignore[unknown-argument]
        "test",
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        data_config=config.data,
    )

    trainer = Seq2SeqTrainer(pad_token_id=target_tokenizer.token_to_id("[PAD]"))
    trainer.train(train_dataloader, val_dataloader, test_dataloader)


def main() -> None:
    """CLI entry point for training seq2seq model."""
    parser = argparse.ArgumentParser(description="Train Seq2Seq model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file (e.g., configs/seq2seq/default.yml)",
    )
    args = parser.parse_args()
    load_config(args.config)
    train_model()


if __name__ == "__main__":
    main()
