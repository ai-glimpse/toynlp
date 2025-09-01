from dataclasses import dataclass, field, asdict
import tyro
from typing import Any


@dataclass
class BertConfig:
    """All in one for everything, without nested dataclasses."""

    # dataset configs
    dataset_path: str = "lucadiliello/bookcorpusopen"
    dataset_name: str | None = None
    batch_size: int = 80  # paper setting: 256
    num_workers: int = 8
    shuffle: bool = True
    # tokenizer configs
    # min_frequency: int = 1
    num_proc: int = 12
    special_tokens: list[str] = field(
        default_factory=lambda: ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"],
    )
    # model configs
    vocab_size: int = 30522
    # model arch configs
    max_seq_length: int = 64  # paper setting: 128, 512
    short_seq_prob: float = 0.1  # probability of creating a short sequence
    masked_lm_prob: float = 0.15  # probability of masking a token
    max_predictions_per_seq: int = 10  # maximum number of masked tokens, paper setting: 20

    d_model: int = 768  # model hidden dimension, paper setting: 768
    attention_d_k: int = 768  # query & key, paper setting: 768
    attention_d_v: int = 768  # value, paper setting: 768
    head_num: int = 12  # paper setting: 12
    d_feed_forward: int = 3072  # paper setting: 3072
    encoder_layers: int = 12  # paper setting: 12

    dropout_ratio: float = 0.0
    # optimizer configs
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    # training configs
    dataset_split_of_tokenizer: str = "train[:10%]"

    # dataset_split_of_model_train: str = "train[:8%]"
    # dataset_split_of_model_val: str = "train[8%:9%]"
    # dataset_split_of_model_test: str = "train[9%:10%]"

    dataset_split_of_model_train: str = "train[8:9]"
    dataset_split_of_model_val: str = "train[7:8]"
    dataset_split_of_model_test: str = "train[9:10]"

    epochs: int = 40
    clip_norm: float | None = None  # Gradient clipping norm, None means no clipping
    # wandb configs
    wandb_name: str | None = None
    wandb_project: str = "Bert"
    wandb_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for logging/serialization."""
        return asdict(self)


def create_config_from_cli() -> BertConfig:
    """Create configuration from command line arguments using tyro."""
    return tyro.cli(BertConfig)


if __name__ == "__main__":
    # Example of using tyro to parse command line arguments
    config = tyro.cli(BertConfig)
    print("Configuration loaded:")
    print(f"  Dataset: {config.dataset_path}")
    print(f"  Max length: {config.max_length}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.epochs}")
