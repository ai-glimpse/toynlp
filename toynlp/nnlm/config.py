from dataclasses import dataclass, field


@dataclass
class OptimizerConfig:
    # name: str = "adam"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    # momentum: float = 0.9
    # beta1: float = 0.9
    # beta2: float = 0.999
    # eps: float = 1e-8
    # warmup_steps: Optional[int] = None


@dataclass
class ModelConfig:
    context_size: int = 6
    vocab_size: int = 17964
    embedding_dim: int = 100
    hidden_dim: int = 60
    # dropout_rate: float = 0.2
    # hidden_sizes: List[int] = field(default_factory=lambda: [512, 256, 128])
    # activation: str = "relu"


@dataclass
class DataConfig:
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True


@dataclass
class TrainingConfig:
    epochs: int = 10
    device: str = "cuda:0"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    early_stopping_patience: int = 5
    # grad_clip: Optional[float] = 1.0


@dataclass
class NNLMConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

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
        training=TrainingConfig(epochs=10, device="cuda:0"),
    )

    print(config)
    print(asdict(config))
