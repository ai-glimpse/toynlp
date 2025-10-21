from datasets import load_dataset, Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, asdict
from typing import Any
from torch.utils.data import DataLoader
from toynlp.gpt.config import GPTConfig
from toynlp.gpt.tokenizer import GPTTokenizer
from toynlp.gpt.model import GPTModel
from toynlp.util import current_device
from toynlp.paths import SST2GPT_MODEL_PATH, GPT_MODEL_PATH
import wandb
from toynlp.util import setup_seed, set_deterministic_mode


setup_seed(1234)  # Set a random seed for reproducibility
set_deterministic_mode()  # Set deterministic mode for reproducibility


@dataclass
class EvaluationConfig:
    with_pretrained: bool = True
    dataset_path: str = "stanfordnlp/sst2"
    test_dataset_path: str = "SetFit/sst2"  # test set labels
    dataset_name: str | None = None
    batch_size: int = 32
    # train
    epochs: int = 10
    # optimizer configs
    learning_rate: float = 2e-5  # paper setting: 6.25e-5
    weight_decay: float = 0.01

    # wandb configs
    wandb_name: str | None = None
    wandb_project: str = "SST2GPT"
    wandb_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for logging/serialization."""
        return asdict(self)

    def __post_init__(self) -> None:
        if self.wandb_name is None:
            self.wandb_name = f"lr({self.learning_rate})_bs({self.batch_size})_withpre({self.with_pretrained})"


evaluation_config = EvaluationConfig()


def get_dataset(
    dataset_path: str,
    dataset_name: str | None,
    split: str,
) -> Dataset:
    dataset = load_dataset(path=dataset_path, name=dataset_name, split=split)
    return dataset  # type: ignore[return-value]


def load_sst2_dataset():
    """Load the SST-2 dataset from the Hugging Face datasets library.

    DatasetDict({
            train: Dataset({
                features: ['idx', 'sentence', 'label'],
                num_rows: 67349
            })
            validation: Dataset({
                features: ['idx', 'sentence', 'label'],
                num_rows: 872
            })
            test: Dataset({
                features: ['idx', 'sentence', 'label'],
                num_rows: 1821
            })
        }).
    """
    return load_dataset("stanfordnlp/sst2")


def collate_fn(batch, max_sequence_length: int = 128) -> dict[str, torch.Tensor]:
    input_ids = []
    token_type_ids = []
    labels = []

    for item in batch:
        encoded = gpt_tokenizer.encode(
            item["sentence"] + " very",
        )
        input_ids.append(torch.tensor(encoded.ids[:max_sequence_length], dtype=torch.long))
        token_type_ids.append(torch.tensor(encoded.type_ids[:max_sequence_length], dtype=torch.long))
        # labels.append(gpt_tokenizer.token_to_id(item["label_text"]))
        labels.append(item["label"])  # 0 or 1
    padding_id = gpt_tokenizer.token_to_id("<pad>")
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=padding_id)  # type: ignore[assignment]
    token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=padding_id)  # type: ignore[assignment]
    labels = torch.tensor(labels)  # type: ignore[assignment]

    return {
        "input_ids": input_ids,  # type: ignore[dict-item]
        "token_type_ids": token_type_ids,  # type: ignore[dict-item]
        "labels": labels,  # type: ignore[dict-item]
    }


def get_split_dataloader(
    dataset_path: str,
    split: str,
    config: GPTConfig,
) -> DataLoader:
    raw_dataset = get_dataset(dataset_path, None, split)  # type: ignore[call-arg]
    if split in {"train", "validation"}:
        # add column label_text: if label=0 -> negative, if label=1 -> positive
        raw_dataset = raw_dataset.map(
            lambda example: {"label_text": "positive" if example["label"] == 1 else "negative", **example},
        )
    if split == "test":
        raw_dataset = raw_dataset.rename_column("text", "sentence")
    dataloader = torch.utils.data.DataLoader(
        raw_dataset.with_format(type="torch"),
        batch_size=evaluation_config.batch_size,
        collate_fn=lambda batch: collate_fn(batch, config.max_seq_length),
    )

    return dataloader


class SST2GPTTrainer:
    def __init__(self, config: GPTConfig) -> None:
        self.config = config
        self.positive_token_id = gpt_tokenizer.token_to_id("positive")
        self.negative_token_id = gpt_tokenizer.token_to_id("negative")
        self.pad_token_id = gpt_tokenizer.token_to_id("<pad>")
        self.model = GPTModel(self.config, padding_idx=self.pad_token_id)
        # load pretrained model if exists
        if evaluation_config.with_pretrained:
            if GPT_MODEL_PATH.exists():
                print(f"Loading pretrained GPT model from {GPT_MODEL_PATH}")
                pretrained_gpt: GPTModel = torch.load(GPT_MODEL_PATH, weights_only=False)
                self.model.load_state_dict(pretrained_gpt.state_dict())
            else:
                print(f"No pretrained GPT model found at {GPT_MODEL_PATH}, training from scratch.")
        self.model_path = SST2GPT_MODEL_PATH
        self.device = current_device
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=evaluation_config.learning_rate,
            weight_decay=evaluation_config.weight_decay,
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.clip_norm = self.config.clip_norm
        if self.clip_norm:
            print(f"Gradient clipping enabled with norm {self.clip_norm}")

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
    ) -> None:
        best_val_acc = 0.0
        for epoch in range(evaluation_config.epochs):
            train_loss, train_acc = self._train_epoch(train_dataloader)
            val_loss, val_acc, test_loss, test_acc = self._validate_epoch(val_dataloader, test_dataloader)

            print(
                f"Epoch {epoch + 1}/{evaluation_config.epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}",
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model, self.model_path)
                print(f"Saved best model({val_loss=:.4f}, {val_acc=:.4f}) from epoch {epoch + 1} to {self.model_path}")

            # log metrics to wandb
            if evaluation_config.wandb_enabled:
                wandb.log(
                    {
                        "TrainLoss": train_loss,
                        "TrainAccuracy": train_acc,
                        "ValLoss": val_loss,
                        "ValAccuracy": val_acc,
                        "TestLoss": test_loss,
                        "TestAccuracy": test_acc,
                        "Epoch": epoch + 1,
                    },
                )

    def _train_epoch(self, train_dataloader: DataLoader) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0

        for batch in train_dataloader:
            self.optimizer.zero_grad()
            input_batch_device, target_batch_device = (
                batch["input_ids"].to(self.device),
                batch["labels"].to(self.device),
            )
            logits = self.model(input_batch_device)
            full_token_logits = logits[:, -1, :]
            # positive, negative token logits
            neg_logits = full_token_logits[:, self.negative_token_id]
            pos_logits = full_token_logits[:, self.positive_token_id]
            label_token_logits = torch.stack((neg_logits, pos_logits), dim=1)
            loss = self.criterion(label_token_logits, target_batch_device)
            loss.backward()
            if self.clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
            self.optimizer.step()

            # Calculate accuracy
            predictions = torch.argmax(label_token_logits, dim=-1)
            correct_predictions += (predictions == target_batch_device).sum().item()

            total_loss += loss.item() * batch["input_ids"].size(0)  # Multiply by batch size
            total_samples += batch["input_ids"].size(0)

        train_loss = total_loss / total_samples
        train_accuracy = correct_predictions / total_samples
        return train_loss, train_accuracy

    def _validate_epoch(
        self, val_dataloader: DataLoader, test_dataloader: DataLoader
    ) -> tuple[float, float, float, float]:
        self.model.eval()
        with torch.no_grad():
            val_loss, val_acc = self.calc_loss_and_accuracy_loader(val_dataloader)
            test_loss, test_acc = self.calc_loss_and_accuracy_loader(test_dataloader)
        return val_loss, val_acc, test_loss, test_acc

    def calc_loss_batch(
        self,
        input_batch: torch.Tensor,
        segment_ids_batch: torch.Tensor,
        target_batch: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.model(input_batch, segment_ids_batch)
        loss = self.criterion(logits, target_batch)
        return loss

    def calc_loss_loader(self, data_loader: DataLoader) -> float:
        total_loss = 0.0
        total_samples = 0  # Track total samples
        for batch in data_loader:
            input_batch_device, segment_ids_batch, target_batch_device = (
                batch["input_ids"].to(self.device),
                batch["token_type_ids"].to(self.device),
                batch["labels"].to(self.device),
            )
            loss = self.calc_loss_batch(input_batch_device, segment_ids_batch, target_batch_device)
            total_loss += loss.item() * batch["input_ids"].size(0)  # Multiply by batch size
            total_samples += batch["input_ids"].size(0)
        return total_loss / total_samples  # Correct average

    def calc_loss_and_accuracy_loader(self, data_loader: DataLoader) -> tuple[float, float]:
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0

        for batch in data_loader:
            input_batch_device, segment_ids_batch, target_batch_device = (
                batch["input_ids"].to(self.device),
                batch["token_type_ids"].to(self.device),
                batch["labels"].to(self.device),
            )
            logits = self.model(input_batch_device, segment_ids_batch)
            predictions = torch.argmax(logits, dim=-1)
            loss = self.criterion(logits, target_batch_device)

            # Calculate accuracy
            correct_predictions += (predictions == target_batch_device).sum().item()

            total_loss += loss.item() * batch["input_ids"].size(0)
            total_samples += batch["input_ids"].size(0)

        avg_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples
        return avg_loss, accuracy


def train_model(config: GPTConfig) -> None:
    if evaluation_config.wandb_enabled:
        wandb.init(
            project=evaluation_config.wandb_project,
            name=evaluation_config.wandb_name,
            config=evaluation_config.to_dict(),
        )
    train_dataloader = get_split_dataloader(
        dataset_path=evaluation_config.dataset_path,  # type: ignore[unknown-argument]
        split="train",
        config=config,
    )
    val_dataloader = get_split_dataloader(
        dataset_path=evaluation_config.dataset_path,  # type: ignore[unknown-argument]
        split="validation",
        config=config,
    )
    test_dataloader = get_split_dataloader(
        evaluation_config.test_dataset_path,  # type: ignore[unknown-argument]
        split="test",
        config=config,
    )
    trainer = SST2GPTTrainer(config=config)
    trainer.train(train_dataloader, val_dataloader, test_dataloader)  # Pass None for test_dataloader


def main(config: GPTConfig) -> None:
    """CLI entry point for training sst2gpt model using tyro configuration."""
    # Load configuration from command line using tyro

    print("=" * 60)
    print("SST2GPT MODEL TRAINING")
    print("=" * 60)
    print(f"Dataset: {evaluation_config.dataset_path}")
    print(f"Training: {evaluation_config.epochs} epochs, lr={evaluation_config.learning_rate}")
    print(f"WandB: {'enabled' if evaluation_config.wandb_enabled else 'disabled'}")
    print("=" * 60)

    train_model(config)


if __name__ == "__main__":
    gpt_tokenizer = GPTTokenizer().load()
    gpt_config = GPTConfig(
        max_seq_length=512,
        vocab_size=40478,
        d_model=768,
        attention_d_k=768,
        attention_d_v=768,
        head_num=12,
        d_feed_forward=3072,
    )

    main(gpt_config)
