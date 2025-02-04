from dataclasses import dataclass, field


@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4


@dataclass
class ModelConfig:
    context_size: int = 6
    vocab_size: int = 20000
    embedding_dim: int = 100
    hidden_dim: int = 60
    dropout_rate: float = 0.2
    with_dropout: bool = True
    with_direct_connection: bool = False


@dataclass
class DataConfig:
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True


@dataclass
class TrainingConfig:
    epochs: int = 10


@dataclass
class WanDbConfig:
    name: str = "unnamed"
    project: str = "NNLM"


@dataclass
class NNLMConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WanDbConfig = field(default_factory=WanDbConfig)

    def __post_init__(self):
        """Basic validation"""
        if self.training.epochs <= 0:
            raise ValueError("Epochs must be positive")
        if self.optimizer.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")


if __name__ == "__main__":
    from dataclasses import asdict

    config = NNLMConfig(
        model=ModelConfig(
            context_size=5,
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
