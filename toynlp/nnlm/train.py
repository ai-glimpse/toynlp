from dataclasses import asdict
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

import wandb
from toynlp.util import current_device
from toynlp.nnlm.config import NNLMConfig, create_config_from_cli
from toynlp.nnlm.model import NNLM
from toynlp.nnlm.tokenizer import NNLMTokenizer


class NNLMTrainer:
    def __init__(self, config: NNLMConfig) -> None:
        self.config = config
        model_config = config.get_model_config()
        self.model = NNLM(model_config)
        self.device = current_device
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    @property
    def model_path(self) -> Path:
        return self.config.model_path

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
    ) -> None:
        best_val_loss = float("inf")
        for epoch in range(self.config.epochs):
            self.model.train()
            for _, batch in enumerate(train_dataloader):
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
                    # torch.save(self.model.state_dict(), "nnlm.pth")
                    torch.save(self.model, self.model_path)

    def calc_loss_batch(self, input_batch: torch.Tensor, target_batch: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits, target_batch)
        return loss

    def calc_loss_loader(self, data_loader: DataLoader) -> float:
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


def get_split_dataloader(
    tokenizer: Tokenizer,
    dataset: Dataset,
    split: str,
    context_size: int = 6,
    batch_size: int = 32,
) -> DataLoader:
    text = " ".join(dataset[split]["text"])
    token_ids = tokenizer.encode(text).ids
    token_ids_tensor = torch.tensor(token_ids).unfold(0, context_size, 1)
    dataloader: DataLoader = DataLoader(token_ids_tensor, batch_size=batch_size, shuffle=True)  # type: ignore[arg-type]
    return dataloader


def run(config: NNLMConfig) -> None:
    wandb.init(
        # set the wandb project where this run will be logged
        project=config.wandb_project,
        name=config.wandb_name,
        # track hyperparameters and run metadata
        config=asdict(config),
    )

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenizer_model_path = config.tokenizer_path
    if not Path(tokenizer_model_path).exists():
        nnlm_tokenizer = NNLMTokenizer(
            model_path=tokenizer_model_path,
            vocab_size=config.vocab_size,
        )
        nnlm_tokenizer.train(dataset["train"])  # type: ignore[unknown-argument]
    tokenizer = NNLMTokenizer(model_path=tokenizer_model_path).load()
    train_dataloader = get_split_dataloader(
        tokenizer,
        dataset,  # type: ignore[unknown-argument]
        "train",
        context_size=config.context_size,
        batch_size=config.batch_size,
    )
    val_dataloader = get_split_dataloader(
        tokenizer,
        dataset,  # type: ignore[unknown-argument]
        "validation",
        context_size=config.context_size,
        batch_size=config.batch_size,
    )
    test_dataloader = get_split_dataloader(
        tokenizer,
        dataset,  # type: ignore[unknown-argument]
        "test",
        context_size=config.context_size,
        batch_size=config.batch_size,
    )

    trainer = NNLMTrainer(config)
    trainer.train(train_dataloader, val_dataloader, test_dataloader)


def main() -> None:
    """CLI entry point for training NNLM model using tyro configuration."""
    # Load configuration from command line using tyro
    config = create_config_from_cli()

    print("=" * 60)
    print("NNLM Training Configuration")
    print("=" * 60)
    print(f"Context size: {config.context_size}")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Embedding dimension: {config.embedding_dim}")
    print(f"Hidden dimension: {config.hidden_dim}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Epochs: {config.epochs}")
    print("=" * 60)

    run(config)


if __name__ == "__main__":
    # Use the new main function with tyro CLI
    main()
