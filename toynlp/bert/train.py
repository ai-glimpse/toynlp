import torch
from torch.utils.data import DataLoader

import wandb
from toynlp.util import current_device
from toynlp.paths import BERT_MODEL_PATH
from toynlp.bert.config import BertConfig, create_config_from_cli
from toynlp.bert.model import BertModel
from toynlp.bert.tokenizer import BertTokenizer
from toynlp.util import setup_seed, set_deterministic_mode
from toynlp.bert.dataset import get_split_dataloader


setup_seed(1234)  # Set a random seed for reproducibility
set_deterministic_mode()  # Set deterministic mode for reproducibility


class BertTrainer:
    def __init__(self, config: BertConfig, pad_token_id: int) -> None:
        self.config = config
        self.model = BertModel(self.config, pad_token_id)
        self.model_path = BERT_MODEL_PATH
        self.device = current_device
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.mlm_criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction="mean")
        self.nsp_criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        self.clip_norm = self.config.clip_norm
        if self.clip_norm:
            print(f"Gradient clipping enabled with norm {self.clip_norm}")

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

            print(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"Train Loss: {train_stats['loss']:.4f}, "
                f"Train MLM Loss: {train_stats['mlm_loss']:.4f}, "
                f"Train NSP Loss: {train_stats['nsp_loss']:.4f}, "
                f"Train NSP Accuracy: {train_stats['nsp_accuracy']:.4f}, "
                f"Val Loss: {val_loss_stats['loss']:.4f}, "
                f"Val MLM Loss: {val_loss_stats['mlm_loss']:.4f}, "
                f"Val NSP Loss: {val_loss_stats['nsp_loss']:.4f}, "
                f"Val NSP Accuracy: {val_loss_stats['nsp_accuracy']:.4f}, "
                f"Test Loss: {test_loss_stats['loss']:.4f}, "
                f"Test NSP Accuracy: {test_loss_stats['nsp_accuracy']:.4f}, "
                f"Test MLM Loss: {test_loss_stats['mlm_loss']:.4f}, "
                f"Test NSP Loss: {test_loss_stats['nsp_loss']:.4f}"
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
                        "ValLoss": val_loss_stats["loss"],
                        "TestLoss": test_loss_stats["loss"],
                        "TrainMLMLoss": train_stats["mlm_loss"],
                        "ValMLMLoss": val_loss_stats["mlm_loss"],
                        "TestMLMLoss": test_loss_stats["mlm_loss"],
                        "TrainNSPLoss": train_stats["nsp_loss"],
                        "ValNSPLoss": val_loss_stats["nsp_loss"],
                        "TestNSPLoss": test_loss_stats["nsp_loss"],
                        "TrainNSPAccuracy": train_stats["nsp_accuracy"],
                        "ValNSPAccuracy": val_loss_stats["nsp_accuracy"],
                        "TestNSPAccuracy": test_loss_stats["nsp_accuracy"],
                        "TrainPerplexity": torch.exp(torch.tensor(train_stats["loss"])),
                        "ValPerplexity": torch.exp(torch.tensor(val_loss_stats["loss"])),
                        "TestPerplexity": torch.exp(torch.tensor(test_loss_stats["loss"])),
                    },
                )

    def _train_epoch(self, train_dataloader: DataLoader) -> dict[str, float]:
        self.model.train()
        total_loss, mlm_loss, nsp_loss = 0.0, 0.0, 0.0
        total_samples = 0
        nsp_total, nsp_correct = 0, 0  # Initialize counters for NSP accuracy
        for batch in train_dataloader:
            self.optimizer.zero_grad()
            batch_input_tokens = batch["tokens"].to(self.device)
            batch_segment_ids = batch["segment_ids"].to(self.device)
            batch_masked_lm_labels = batch["masked_lm_labels"].to(self.device)
            batch_is_random_next = batch["is_random_next"].to(self.device)
            loss_stats = self.calc_loss_batch(
                batch_input_tokens, batch_segment_ids, batch_is_random_next, batch_masked_lm_labels
            )
            loss: torch.Tensor = loss_stats["loss"]

            nsp_total += loss_stats["nsp_total"]
            nsp_correct += loss_stats["nsp_correct"]
            loss.backward()
            if self.clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
            self.optimizer.step()
            batch_size = batch_input_tokens.size(0)
            total_loss += loss.item() * batch_size  # Multiply by batch size
            mlm_loss += loss_stats["mlm_loss"].item() * batch_size
            nsp_loss += loss_stats["nsp_loss"].item() * batch_size
            total_samples += batch_size
        avg_train_loss = total_loss / total_samples
        avg_mlm_loss = mlm_loss / total_samples
        avg_nsp_loss = nsp_loss / total_samples
        train_nsp_accuracy = nsp_correct / nsp_total if nsp_total > 0 else 0

        train_stats = {
            "loss": avg_train_loss,
            "mlm_loss": avg_mlm_loss,
            "nsp_loss": avg_nsp_loss,
            "nsp_accuracy": train_nsp_accuracy,
        }

        return train_stats

    def _validate_epoch(
        self,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
    ) -> tuple[dict[str, float], dict[str, float]]:
        self.model.eval()
        with torch.no_grad():
            val_loss_stats = self.calc_loss_loader(val_dataloader)
            test_loss_stats = self.calc_loss_loader(test_dataloader)
        return val_loss_stats, test_loss_stats

    def calc_loss_batch(
        self,
        batch_input_tokens: torch.Tensor,
        batch_segment_ids: torch.Tensor,
        batch_is_random_next: torch.Tensor,
        batch_masked_lm_labels: torch.Tensor,
    ) -> dict[str, float | torch.Tensor]:
        nsp_logits_output, mlm_logits_output = self.model(batch_input_tokens, batch_segment_ids)
        pred = mlm_logits_output.reshape(-1, mlm_logits_output.shape[-1])
        target_batch = batch_masked_lm_labels.reshape(-1)
        # print(
        #     f"percetage of mask: {(batch_masked_lm_labels != 0).sum(dim=1) / batch_input_tokens.size(1)}"
        # )  # should be ~15% of tokens
        mlm_loss = self.mlm_criterion(pred, target_batch)
        nsp_loss = self.nsp_criterion(nsp_logits_output, batch_is_random_next)
        loss = mlm_loss + nsp_loss

        # nsp total and correct
        nsp_total = batch_is_random_next.size(0)
        nsp_correct = (nsp_logits_output.argmax(dim=-1) == batch_is_random_next).sum().item()

        stats = {
            "loss": loss,
            "mlm_loss": mlm_loss,
            "nsp_loss": nsp_loss,
            "nsp_total": nsp_total,
            "nsp_correct": nsp_correct,
        }
        return stats

    def calc_loss_loader(self, data_loader: DataLoader) -> dict[str, float]:
        total_loss, nsp_loss, mlm_loss = 0.0, 0.0, 0.0
        total_samples = 0  # Track total samples

        nsp_total, nsp_correct = 0, 0  # Initialize counters for NSP accuracy
        for batch in data_loader:
            input_batch_device = batch["tokens"].to(self.device)
            batch_size = input_batch_device.size(0)
            segment_batch_device = batch["segment_ids"].to(self.device)
            is_random_next_batch_device = batch["is_random_next"].to(self.device)
            target_batch_device = batch["masked_lm_labels"].to(self.device)
            loss_stats = self.calc_loss_batch(
                input_batch_device, segment_batch_device, is_random_next_batch_device, target_batch_device
            )
            total_loss += loss_stats["loss"].item() * batch_size  # Multiply by batch size
            mlm_loss += loss_stats["mlm_loss"].item() * batch_size
            nsp_loss += loss_stats["nsp_loss"].item() * batch_size
            total_samples += batch_size

            nsp_total += loss_stats["nsp_total"]
            nsp_correct += loss_stats["nsp_correct"]

        avg_loss = total_loss / total_samples  # Correct average
        mlm_loss = mlm_loss / total_samples if total_samples > 0 else 0
        nsp_loss = nsp_loss / total_samples if total_samples > 0 else 0
        nsp_accuracy = nsp_correct / nsp_total if nsp_total > 0 else 0
        stats = {
            "loss": avg_loss,
            "mlm_loss": mlm_loss,
            "nsp_loss": nsp_loss,
            "nsp_accuracy": nsp_accuracy,
        }

        return stats


def train_model(config: BertConfig) -> None:
    """Train the BERT model with the given configuration."""
    if config.wandb_enabled:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_name,
            config=config.to_dict(),
        )

    tokenizer = BertTokenizer().load()

    train_dataloader = get_split_dataloader(
        config.dataset_path,
        config.dataset_split_of_model_train,
        config=config,
    )
    val_dataloader = get_split_dataloader(
        config.dataset_path,
        config.dataset_split_of_model_val,
        config=config,
    )
    test_dataloader = get_split_dataloader(
        config.dataset_path,
        config.dataset_split_of_model_test,
        config=config,
    )

    trainer = BertTrainer(config=config, pad_token_id=tokenizer.token_to_id("[PAD]"))
    trainer.train(train_dataloader, val_dataloader, test_dataloader)


def main() -> None:
    """CLI entry point for training BERT model using tyro configuration."""
    # Load configuration from command line using tyro
    config = create_config_from_cli()

    print("=" * 60)
    print("BERT MODEL TRAINING")
    print("=" * 60)
    print(f"Dataset: {config.dataset_path}")
    print(f"Training: {config.epochs} epochs, lr={config.learning_rate}")
    print(f"WandB: {'enabled' if config.wandb_enabled else 'disabled'}")
    print("=" * 60)

    train_model(config)


if __name__ == "__main__":
    main()
