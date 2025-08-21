from dataclasses import dataclass, field, asdict
import tyro
from typing import Any


@dataclass
class TransformerConfig:
    """All in one for everything, without nested dataclasses."""

    # dataset configs
    dataset_path: str = "bentrevett/multi30k"
    dataset_name: str | None = None
    source_lang: str = "de"
    target_lang: str = "en"
    max_length: int = 1000
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    # tokenizer configs
    min_frequency: int = 1
    num_proc: int = 8
    special_tokens: list[str] = field(
        default_factory=lambda: ["[UNK]", "[BOS]", "[EOS]", "[PAD]"],
    )
    # model configs
    vocab_size: int = 30000
    # model arch configs
    max_source_seq_length: int = 1000
    max_target_seq_length: int = 1000

    # For each of these we use dk = dv = dmodel/h = 64
    d_model: int = 256  # model hidden dimension, paper setting: 512
    attention_d_k: int = 256  # query & key, paper setting: 512
    attention_d_v: int = 256  # value, paper setting: 512
    # we employ h = 8 parallel attention layers, or heads
    head_num: int = 4  # paper setting: 8
    d_feed_forward: int = 1024  # paper setting: 2048
    encoder_layers: int = 6  # paper setting: 6
    decoder_layers: int = 6  # paper setting: 6

    dropout_ratio: float = 0.5
    teacher_forcing_ratio: float = 0.5
    # optimizer configs
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    # training configs
    epochs: int = 20
    clip_norm: float | None = None  # Gradient clipping norm, None means no clipping
    # inference configs
    inference_max_length: int = 50
    # evaluation configs
    evaluation_max_samples: int | None = None
    evaluation_batch_size: int = 32
    # wandb configs
    wandb_name: str | None = None
    wandb_project: str = "Transformer"
    wandb_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for logging/serialization."""
        return asdict(self)


def create_config_from_cli() -> TransformerConfig:
    """Create configuration from command line arguments using tyro."""
    return tyro.cli(TransformerConfig)


if __name__ == "__main__":
    # Example of using tyro to parse command line arguments
    config = tyro.cli(TransformerConfig)
    print("Configuration loaded:")
    print(f"  Dataset: {config.dataset_path}")
    print(f"  Source language: {config.source_lang}")
    print(f"  Target language: {config.target_lang}")
    print(f"  Max length: {config.max_length}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.epochs}")
