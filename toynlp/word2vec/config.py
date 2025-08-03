from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DatasetConfig:
    path: str = "Salesforce/wikitext"
    name: str = "wikitext-103-raw-v1"


@dataclass
class DataConfig:
    # token processing
    cbow_n_words: int = 4
    skip_gram_n_words: int = 4

    # data loader
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True


@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4


@dataclass
class ModelConfig:
    vocab_size: int = 20000
    embedding_dim: int = 256


@dataclass
class TrainingConfig:
    epochs: int = 10


@dataclass
class WanDbConfig:
    name: str | None = None
    project: str = "Word2Vec"


@dataclass
class Word2VecConfig:
    model_name: Literal["cbow", "skip_gram"] = "cbow"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WanDbConfig = field(default_factory=WanDbConfig)

    def __post_init__(self) -> None:
        """Basic validation."""
        if self.training.epochs <= 0:
            raise ValueError("Epochs must be positive")
        if self.optimizer.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        if self.wandb.name is None:
            self.wandb.name = self._get_wandb_name()

    def _get_wandb_name(self) -> str:
        s = f"[{self.model_name}]embedding_dim:{self.model.embedding_dim}"
        return s


if __name__ == "__main__":
    from dataclasses import asdict

    config = Word2VecConfig(
        model=ModelConfig(
            embedding_dim=256,
        ),
        optimizer=OptimizerConfig(
            learning_rate=5e-5,
        ),
        data=DataConfig(
            batch_size=64,
        ),
        training=TrainingConfig(epochs=10),
    )

    print(config)
    print(asdict(config))
