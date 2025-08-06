from dataclasses import asdict
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

import wandb
from toynlp.util import current_device
from toynlp.paths import CBOW_MODEL_PATH, SKIP_GRAM_MODEL_PATH, W2V_TOKENIZER_PATH
from toynlp.word2vec.config import (
    DataConfig,
    DatasetConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
    Word2VecConfig,
)
from toynlp.word2vec.dataset import get_split_dataloader
from toynlp.word2vec.model import CbowModel, SkipGramModel
from toynlp.word2vec.tokenizer import Word2VecTokenizer


class Word2VecTrainer:
    def __init__(self, config: Word2VecConfig) -> None:
        self.config = config
        if self.config.model_name == "cbow":
            self.model = CbowModel(config.model)
            self.model_path = CBOW_MODEL_PATH
        else:
            self.model = SkipGramModel(config.model)  # type: ignore[assignment]
            self.model_path = SKIP_GRAM_MODEL_PATH
        self.device = current_device
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
        )

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
        logits = self.model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits, target_batch)
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


def train_tokenizer(config: Word2VecConfig) -> None:
    if not Path(W2V_TOKENIZER_PATH).exists():
        word2vec_tokenizer = Word2VecTokenizer(
            vocab_size=config.model.vocab_size,
        )
        dataset = load_dataset(path=config.dataset.path, name=config.dataset.name)
        word2vec_tokenizer.train(dataset["train"])  # type: ignore[unknown-argument]
    else:
        print(f"Tokenizer already exists at {W2V_TOKENIZER_PATH}")


def train_model(config: Word2VecConfig) -> None:
    model_name = config.model_name
    wandb.init(
        # set the wandb project where this run will be logged
        project=config.wandb.project,
        name=config.wandb.name,
        # track hyperparameters and run metadata
        config=asdict(config),
    )
    dataset = load_dataset(path=config.dataset.path, name=config.dataset.name)
    tokenizer = Word2VecTokenizer().load()
    train_dataloader = get_split_dataloader(
        dataset,  # type: ignore[unknown-argument]
        "train",
        tokenizer=tokenizer,
        data_config=config.data,
        model_name=model_name,
    )
    val_dataloader = get_split_dataloader(
        dataset,  # type: ignore[unknown-argument]
        "validation",
        tokenizer=tokenizer,
        data_config=config.data,
        model_name=model_name,
    )
    test_dataloader = get_split_dataloader(
        dataset,  # type: ignore[unknown-argument]
        "test",
        tokenizer=tokenizer,
        data_config=config.data,
        model_name=model_name,
    )

    trainer = Word2VecTrainer(config)
    trainer.train(train_dataloader, val_dataloader, test_dataloader)


def train_cbow():
    config = Word2VecConfig(
        model_name="cbow",
        dataset=DatasetConfig(
            path="Salesforce/wikitext",
            name="wikitext-103-raw-v1",
        ),
        model=ModelConfig(
            embedding_dim=100,
        ),
        optimizer=OptimizerConfig(
            learning_rate=0.01,
        ),
        data=DataConfig(
            batch_size=512,
            num_workers=8,
        ),
        training=TrainingConfig(epochs=10),
    )

    train_tokenizer(config)
    train_model(config)


def train_skip_gram():
    config = Word2VecConfig(
        model_name="skip_gram",
        dataset=DatasetConfig(
            path="Salesforce/wikitext",
            name="wikitext-103-raw-v1",
        ),
        model=ModelConfig(
            embedding_dim=100,
        ),
        optimizer=OptimizerConfig(
            learning_rate=0.01,
        ),
        data=DataConfig(
            batch_size=64,
            num_workers=8,
        ),
        training=TrainingConfig(epochs=5),
    )

    train_tokenizer(config)
    train_model(config)


if __name__ == "__main__":
    # train_cbow()
    train_skip_gram()
