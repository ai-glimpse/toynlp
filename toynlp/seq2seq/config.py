from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    path: str = "wmt/wmt14"
    name: str = "fr-en"


@dataclass
class DataConfig:
    # max length of the input and output sequences
    max_length: int = 1000
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
    source_vocab_size: int = 80000
    target_vocab_size: int = 80000

    embedding_dim: int = 1000
    hidden_dim: int = 1000
    num_layers: int = 4

    teacher_forcing_ratio: float = 0.5


@dataclass
class TrainingConfig:
    epochs: int = 10


@dataclass
class WanDbConfig:
    name: str | None = None
    project: str = "Seq2Seq"


@dataclass
class Seq2SeqConfig:
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
        s = f"E:{self.model.embedding_dim},H:{self.model.hidden_dim},L:{self.model.num_layers}"
        return s


if __name__ == "__main__":
    from dataclasses import asdict

    config = Seq2SeqConfig(
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
