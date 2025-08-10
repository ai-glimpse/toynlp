import pathlib
from dataclasses import dataclass, asdict
from typing import Any
import tyro

from toynlp.paths import _MODEL_PATH


@dataclass
class NNLMConfig:
    """All in one for everything, without nested dataclasses."""

    # model configs
    context_size: int = 6
    vocab_size: int = 20000
    embedding_dim: int = 100
    hidden_dim: int = 60
    dropout_rate: float = 0.2
    with_dropout: bool = True
    with_direct_connection: bool = False

    # optimizer configs
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # data configs
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True

    # training configs
    epochs: int = 10

    # wandb configs
    wandb_name: str | None = None
    wandb_project: str = "NNLM"

    # path configs
    model_path: pathlib.Path = _MODEL_PATH / "nnlm" / "model.pt"
    tokenizer_path: pathlib.Path = _MODEL_PATH / "nnlm" / "tokenizer.json"

    def __post_init__(self) -> None:
        """Basic validation and path setup."""
        if self.epochs <= 0:
            raise ValueError("Epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        if self.wandb_name is None:
            self.wandb_name = self._get_wandb_name()

        # Ensure paths are absolute
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_wandb_name(self) -> str:
        """Fields: hidden_dim, with_dropout, dropout_rate, with_direct_connection."""
        s = f"hidden_dim:{self.hidden_dim};with_direct_connection:{self.with_direct_connection}"
        if self.with_dropout:
            s += f";dropout:{self.dropout_rate}"
        else:
            s += ";no_dropout"
        return s

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for logging/serialization."""
        return asdict(self)


def create_config_from_cli() -> NNLMConfig:
    """Create configuration from command line arguments using tyro."""
    return tyro.cli(NNLMConfig)


if __name__ == "__main__":
    # Example of using tyro to parse command line arguments
    config = tyro.cli(NNLMConfig)
    print("Configuration loaded:")
    print(f"  Context size: {config.context_size}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Embedding dimension: {config.embedding_dim}")
    print(f"  Hidden dimension: {config.hidden_dim}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.epochs}")
