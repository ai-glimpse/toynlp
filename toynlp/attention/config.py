from dataclasses import dataclass, field, asdict
import tyro
from typing import Any



@dataclass
class AttentionConfig:
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
    source_vocab_size: int = 8000
    target_vocab_size: int = 6000
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2
    dropout_ratio: float = 0.5
    teacher_forcing_ratio: float = 0.5
    # optimizer configs
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    # training configs
    epochs: int = 10
    clip_norm: float | None = None  # Gradient clipping norm, None means no clipping
    # inference configs
    inference_max_length: int = 50
    # evaluation configs
    evaluation_max_samples: int | None = None
    evaluation_batch_size: int = 32
    # wandb configs
    wandb_name: str | None = None
    wandb_project: str = "Attention"
    wandb_enabled: bool = True


    def get_lang_vocab_size(self, lang: str) -> int:
        """Get vocabulary size for a specific language."""
        if lang == self.source_lang:
            return self.source_vocab_size
        if lang == self.target_lang:
            return self.target_vocab_size
        msg = f"Language '{lang}' not supported. Use '{self.source_lang}' or '{self.target_lang}'"
        raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for logging/serialization."""
        return asdict(self)


# Global configuration instance
_config: AttentionConfig | None = None


def get_config() -> AttentionConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = AttentionConfig()
    return _config


def set_config(config: AttentionConfig) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def load_config_from_cli() -> AttentionConfig:
    """Load configuration from command line arguments using tyro."""
    config = tyro.cli(AttentionConfig)
    set_config(config)
    return config


def reset_config() -> None:
    """Reset the global configuration to default values."""
    global _config
    _config = AttentionConfig()


if __name__ == "__main__":
    # Example of using tyro to parse command line arguments
    config = tyro.cli(AttentionConfig)
    print("Configuration loaded:")
    print(f"  Dataset: {config.dataset_path}")
    print(f"  Source language: {config.source_lang}")
    print(f"  Target language: {config.target_lang}")
    print(f"  Embedding dimension: {config.embedding_dim}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.epochs}")
