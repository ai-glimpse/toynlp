from dataclasses import dataclass, field, asdict
import tyro
from typing import Any


@dataclass
class GPTConfig:
    """All in one for everything, without nested dataclasses."""

    # dataset configs
    dataset_path: str = "lucadiliello/bookcorpusopen"
    dataset_name: str | None = None
    batch_size: int = 16  # paper setting: 64
    num_workers: int = 8
    shuffle: bool = True
    # tokenizer configs
    # min_frequency: int = 1
    num_proc: int = 8
    # model configs
    vocab_size: int = 40478  # paper: (BPE) vocabulary with 40,478 merges
    special_tokens: list[str] = field(
        # eod: end of document; bos: begin of sequence; eos: end of sequence
        default_factory=lambda: ["<bos>", "<eos>", "<unk>", "<pad>"],
    )
    # model arch configs
    max_seq_length: int = 512  # paper setting: 128, 512

    d_model: int = 768  # model hidden dimension, paper setting: 768
    attention_d_k: int = 768  # query & key, paper setting: 768
    attention_d_v: int = 768  # value, paper setting: 768
    head_num: int = 12  # paper setting: 12
    d_feed_forward: int = 3072  # paper setting: 3072
    decoder_layers: int = 12  # paper setting: 12

    dropout_ratio: float = 0.1  # paper setting: 0.1

    # optimizer configs
    learning_rate: float = 2.5e-4  # paper setting: 2.5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200  # paper setting: 2000
    # training configs
    # dataset_split_of_tokenizer: str = "train[:90%]"
    # dataset_split_of_model_train: str = "train[:90%]"
    # dataset_split_of_model_val: str = "train[90%:95%]"
    # dataset_split_of_model_test: str = "train[95%:]"

    dataset_split_of_tokenizer: str = "train[:1%]"
    dataset_split_of_model_train: str = "train[:8]"
    dataset_split_of_model_val: str = "train[8:9]"
    dataset_split_of_model_test: str = "train[9:10]"

    epochs: int = 100
    clip_norm: float | None = 1.0  # Gradient clipping norm, None means no clipping
    # wandb configs
    wandb_name: str | None = None
    wandb_project: str = "GPT"
    wandb_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for logging/serialization."""
        return asdict(self)


def create_config_from_cli() -> GPTConfig:
    """Create configuration from command line arguments using tyro."""
    return tyro.cli(GPTConfig)


if __name__ == "__main__":
    # Example of using tyro to parse command line arguments
    config = tyro.cli(GPTConfig)
    print("Configuration loaded:")
    print(f"  Dataset: {config.dataset_path}")
    print(f"  Max length: {config.max_seq_length}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.epochs}")
