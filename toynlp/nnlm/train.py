from dataclasses import asdict

import torch
import wandb
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader

from toynlp.nnlm.config import (
    DataConfig,
    ModelConfig,
    NNLMConfig,
    OptimizerConfig,
    TrainingConfig,
)
from toynlp.nnlm.model import NNLM
from toynlp.nnlm.tokenizer import nnlm_tokenizer


def get_split_dataloader(
    dataset: Dataset,
    split: str,
    context_size: int = 6,
    batch_size: int = 32,
) -> DataLoader:
    text = " ".join(dataset[split]["text"])
    token_ids = nnlm_tokenizer.encode(text).ids
    token_ids_tensor = torch.tensor(token_ids).unfold(0, context_size, 1)
    dataloader = DataLoader(token_ids_tensor, batch_size=batch_size, shuffle=True)
    return dataloader


class NNLMTrainer:
    def __init__(self, config: NNLMConfig):
        self.config = config
        self.model = NNLM(
            vocab_size=config.model.vocab_size, seq_len=config.model.context_size
        )
        self.device = torch.device(config.training.device)
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
    ):
        for epoch in range(self.config.training.epochs):
            self.model.train()
            for i, batch in enumerate(train_dataloader):
                input_batch, target_batch = batch[:, :-1], batch[:, -1]
                input_batch, target_batch = (
                    input_batch.to(self.device),
                    target_batch.to(self.device),
                )
                self.optimizer.zero_grad()
                loss = self.calc_loss_batch(input_batch, target_batch)
                loss.backward()
                self.optimizer.step()
                # print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}")

            self.model.eval()
            with torch.no_grad():
                train_loss = self.calc_loss_loader(train_dataloader)
                val_loss = self.calc_loss_loader(val_dataloader)
                test_loss = self.calc_loss_loader(test_dataloader)
                print(
                    f"Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}, Test Loss: {test_loss}"
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
                    }
                )

    def calc_loss_batch(self, input_batch, target_batch):
        logits = self.model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits, target_batch)
        return loss

    def calc_loss_loader(self, data_loader):
        total_loss = 0.0
        total_samples = 0  # Track total samples
        for batch in data_loader:
            input_batch, target_batch = batch[:, :-1], batch[:, -1]
            input_batch, target_batch = (
                input_batch.to(self.device),
                target_batch.to(self.device),
            )
            loss = self.calc_loss_batch(input_batch, target_batch)
            total_loss += loss.item() * input_batch.size(0)  # Multiply by batch size
            total_samples += input_batch.size(0)
        return total_loss / total_samples  # Correct average


def run(config: NNLMConfig):
    wandb.init(
        # set the wandb project where this run will be logged
        project="NNLM",
        # track hyperparameters and run metadata
        config=asdict(config),
    )

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_dataloader = get_split_dataloader(
        dataset,
        "train",
        context_size=config.model.context_size,
        batch_size=config.data.batch_size,
    )
    val_dataloader = get_split_dataloader(
        dataset,
        "validation",
        context_size=config.model.context_size,
        batch_size=config.data.batch_size,
    )
    test_dataloader = get_split_dataloader(
        dataset,
        "test",
        context_size=config.model.context_size,
        batch_size=config.data.batch_size,
    )

    trainer = NNLMTrainer(config)
    trainer.train(train_dataloader, val_dataloader, test_dataloader)


if __name__ == "__main__":
    config = NNLMConfig(
        model=ModelConfig(
            context_size=6,
            vocab_size=20000,
            embedding_dim=100,
            hidden_dim=30,
        ),
        optimizer=OptimizerConfig(
            learning_rate=1e-4,
            weight_decay=1e-4,
        ),
        data=DataConfig(
            batch_size=128,
        ),
        training=TrainingConfig(epochs=10, device="cuda:0"),
    )
    run(config)
