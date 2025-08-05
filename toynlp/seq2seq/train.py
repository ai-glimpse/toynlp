from dataclasses import asdict

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

import wandb
from toynlp.device import current_device
from toynlp.paths import SEQ2SEQ_MODEL_PATH
from toynlp.seq2seq.config import (
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
    Seq2SeqConfig,
)
from toynlp.seq2seq.dataset import get_split_dataloader
from toynlp.seq2seq.model import Seq2SeqModel
from toynlp.seq2seq.tokenizer import Seq2SeqTokenizer


class Seq2SeqTrainer:
    def __init__(self, config: Seq2SeqConfig, pad_token_id: int) -> None:
        self.config = config
        self.model = Seq2SeqModel(config.model)
        self.model_path = SEQ2SEQ_MODEL_PATH
        self.device = current_device
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
        )
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)

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
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model, self.model_path)
                print(f"Saved best model to {self.model_path}")

            print(
                f"Epoch {epoch + 1}/{self.config.training.epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Test Loss: {test_loss:.4f}",
            )

            # log metrics to wandb
            wandb.log(
                {
                    "TrainLoss": train_loss,
                    "Val Loss": val_loss,
                    "Test Loss": test_loss,
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
        # TODO: make it clear!
        logits = self.model(input_batch, target_batch)
        # print(f"logits shape: {logits.shape}, target_batch shape: {target_batch.shape}")
        pred = logits[:, 1:, :].reshape(-1, logits.shape[-1])
        target_batch = target_batch[:, 1:].reshape(-1)
        # print(f"target_batch min: {target_batch.min().item()}, max: {target_batch.max().item()}")
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


def train_model(config: Seq2SeqConfig) -> None:
    wandb.init(
        # set the wandb project where this run will be logged
        project=config.wandb.project,
        name=config.wandb.name,
        # track hyperparameters and run metadata
        config=asdict(config),
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

    trainer = Seq2SeqTrainer(config=config, pad_token_id=target_tokenizer.token_to_id("[PAD]"))
    trainer.train(train_dataloader, val_dataloader, test_dataloader)


def train():
    config = Seq2SeqConfig(
        model=ModelConfig(
            embedding_dim=256,
            hidden_dim=512,
            num_layers=2,
        ),
        optimizer=OptimizerConfig(
            learning_rate=0.005,
        ),
        data=DataConfig(
            batch_size=256,
            num_workers=8,
        ),
        training=TrainingConfig(epochs=100),
    )

    train_model(config)


if __name__ == "__main__":
    train()
