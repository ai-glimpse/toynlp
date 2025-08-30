from dataclasses import dataclass, field, asdict
import tyro
from typing import Any


@dataclass
class BertConfig:
    """All in one for everything, without nested dataclasses."""

    # dataset configs
    dataset_path: str = "lucadiliello/bookcorpusopen"
    dataset_name: str | None = None
    max_length: int = 1000
    batch_size: int = 256
    num_workers: int = 8
    shuffle: bool = True
    # tokenizer configs
    # min_frequency: int = 1
    num_proc: int = 12
    special_tokens: list[str] = field(
        default_factory=lambda: ["[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]"],
    )
    # model configs
    vocab_size: int = 30522
    # model arch configs
    max_seq_length: int = 1000

    # For each of these we use dk = dv = dmodel/h = 64
    d_model: int = 256  # model hidden dimension, paper setting: 768
    attention_d_k: int = 256  # query & key, paper setting: 768
    attention_d_v: int = 256  # value, paper setting: 768
    # we employ h = 8 parallel attention layers, or heads
    head_num: int = 4  # paper setting: 12
    d_feed_forward: int = 512  # paper setting: 2048
    encoder_layers: int = 6  # paper setting: 12

    dropout_ratio: float = 0.1
    # optimizer configs
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    # training configs
    epochs: int = 20
    clip_norm: float | None = 1.0  # Gradient clipping norm, None means no clipping
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
