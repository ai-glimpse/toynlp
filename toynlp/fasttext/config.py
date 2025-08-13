from dataclasses import dataclass, field, asdict
import tyro


@dataclass
class FastTextConfig:
    """All in one for everything, without nested dataclasses."""

    # dataset configs
    dataset_path: str = "stanfordnlp/imdb"
    dataset_name: str | None = None
    training_percentage: float = 70

    max_length: int = 2500
    batch_size: int = 512
    num_workers: int = 4
    shuffle: bool = True
    # tokenizer configs
    min_frequency: int = 1
    num_proc: int = 8
    special_tokens: list[str] = field(
        default_factory=lambda: ["[UNK]", "[PAD]"],
    )
    # model configs
    vocab_size: int = 100000
    num_classes: int = 2
    hidden_dim: int = 10
    embedding_dim: int = 300
    dropout_ratio: float = 0.8
    # optimizer configs
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    # training configs
    epochs: int = 30
    clip_norm: float | None = None  # Gradient clipping norm, None means no clipping
    # wandb configs
    wandb_name: str | None = None
    wandb_project: str = "FastText"
    wandb_enabled: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def create_config_from_cli() -> FastTextConfig:
    """Create configuration from command line arguments using tyro."""
    return tyro.cli(FastTextConfig)


if __name__ == "__main__":
    # Example of using tyro to parse command line arguments
    config = tyro.cli(FastTextConfig)
    print("Configuration loaded:")
    print(f"  Dataset: {config.dataset_path}")
    print(f"  Embedding dimension: {config.embedding_dim}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.epochs}")
