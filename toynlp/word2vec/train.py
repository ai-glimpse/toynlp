from dataclasses import asdict
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

import wandb
from toynlp.device import current_device
from toynlp.word2vec.config import (
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
    Word2VecConfig,
)
from toynlp.word2vec.dataset import get_split_dataloader
from toynlp.word2vec.model import Word2VecModel
from toynlp.word2vec.tokenizer import Word2VecTokenizer


class Word2VecTrainer:
    def __init__(self, config: Word2VecConfig) -> None:
        self.config = config
        self.model = Word2VecModel(config.model)
        self.device = current_device
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
        )

    @property
    def model_path(self) -> Path:
        return self.config.paths.model_path

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
    ) -> None:
        best_val_loss = float("inf")
        for epoch in range(self.config.training.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            for input_batch, target_batch in train_dataloader:
                input_batch_device, target_batch_device = (
                    input_batch.to(self.device),
                    target_batch.to(self.device),
                )
                loss = self.calc_loss_batch(input_batch_device, target_batch_device)
                loss.backward()
                self.optimizer.step()
                # print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}")

            self.model.eval()
            with torch.no_grad():
                train_loss = self.calc_loss_loader(train_dataloader)
                val_loss = self.calc_loss_loader(val_dataloader)
                test_loss = self.calc_loss_loader(test_dataloader)
                print(
                    f"Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}, Test Loss: {test_loss}",
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

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model, self.model_path)

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
    path_config = config.paths
    tokenizer_model_path = path_config.tokenizer_path

    if not Path(tokenizer_model_path).exists():
        word2vec_tokenizer = Word2VecTokenizer(
            model_path=tokenizer_model_path,
            vocab_size=config.model.vocab_size,
        )
        dataset = load_dataset(path=config.dataset.path, name=config.dataset.name)
        word2vec_tokenizer.train(dataset["train"])  # type: ignore[unknown-argument]
    else:
        print(f"Tokenizer already exists at {tokenizer_model_path}")


def train_model(config: Word2VecConfig) -> None:
    wandb.init(
        # set the wandb project where this run will be logged
        project=config.wandb.project,
        name=config.wandb.name,
        # track hyperparameters and run metadata
        config=asdict(config),
    )
    dataset = load_dataset(path=config.dataset.path, name=config.dataset.name)
    tokenizer_model_path = config.paths.tokenizer_path
    tokenizer = Word2VecTokenizer(model_path=tokenizer_model_path).load()
    train_dataloader = get_split_dataloader(
        dataset,  # type: ignore[unknown-argument]
        "train",
        tokenizer=tokenizer,
        data_config=config.data,
    )
    val_dataloader = get_split_dataloader(
        dataset,  # type: ignore[unknown-argument]
        "validation",
        tokenizer=tokenizer,
        data_config=config.data,
    )
    test_dataloader = get_split_dataloader(
        dataset,  # type: ignore[unknown-argument]
        "validation",
        tokenizer=tokenizer,
        data_config=config.data,
    )

    trainer = Word2VecTrainer(config)
    trainer.train(train_dataloader, val_dataloader, test_dataloader)


if __name__ == "__main__":
    config = Word2VecConfig(
        model=ModelConfig(
            embedding_dim=256,
        ),
        optimizer=OptimizerConfig(
            learning_rate=1e-4,
            weight_decay=1e-4,
        ),
        data=DataConfig(
            batch_size=512,
            num_workers=8,
        ),
        training=TrainingConfig(epochs=5),
    )

    train_tokenizer(config)
    train_model(config)
