import yaml
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TokenizerConfig:
    min_frequency: int = 1
    num_proc: int = 8
    special_tokens: list[str] = field(
        default_factory=lambda: ["[UNK]", "[BOS]", "[EOS]", "[PAD]"],
    )

    def __post_init__(self) -> None:
        if not isinstance(self.special_tokens, list):
            self.special_tokens = list(self.special_tokens)


@dataclass
class DatasetConfig:
    path: str = "bentrevett/multi30k"
    name: str | None = None
    source_lang: str = "de"
    target_lang: str = "en"

    max_length: int = 1000
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True


@dataclass
class DataConfig:
    max_length: int = 1000
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True


@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4


@dataclass
class ModelConfig:
    source_lang: str = "de"
    target_lang: str = "en"
    source_vocab_size: int = 8000
    target_vocab_size: int = 6000
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 2
    dropout_ratio: float = 0.5
    teacher_forcing_ratio: float = 0.5


@dataclass
class TrainingConfig:
    epochs: int = 10
    clip_norm: float | None = None  # Gradient clipping norm, None means no clipping


@dataclass
class InferenceConfig:
    max_length: int = 50


@dataclass
class EvaluationConfig:
    max_samples: int | None = None
    batch_size: int = 32


@dataclass
class WanDbConfig:
    name: str | None = None
    project: str = "Seq2Seq"
    enabled: bool = True


@dataclass
class Seq2SeqConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    wandb: WanDbConfig = field(default_factory=WanDbConfig)

    def __post_init__(self) -> None:
        if self.training.epochs <= 0:
            raise ValueError("Epochs must be positive")
        if self.optimizer.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.wandb.name is None:
            self.wandb.name = self._get_wandb_name()

    def get_lang_vocab_size(self, lang: str) -> int:
        if lang == self.dataset.source_lang:
            return self.model.source_vocab_size
        if lang == self.dataset.target_lang:
            return self.model.target_vocab_size
        msg = f"Language '{lang}' not supported. Use '{self.dataset.source_lang}' or '{self.dataset.target_lang}'"
        raise ValueError(msg)

    def _get_wandb_name(self) -> str:
        return f"E:{self.model.embedding_dim},H:{self.model.hidden_dim},L:{self.model.num_layers}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Seq2SeqConfig":
        # Convert nested dictionaries to their respective dataclass types
        kwargs = {}
        config_classes = {
            "dataset": DatasetConfig,
            "model": ModelConfig,
            "optimizer": OptimizerConfig,
            "training": TrainingConfig,
            "inference": InferenceConfig,
            "evaluation": EvaluationConfig,
            "wandb": WanDbConfig,
            "tokenizer": TokenizerConfig,
        }

        for key, value in data.items():
            if isinstance(value, dict) and key in config_classes:
                kwargs[key] = config_classes[key](**value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    def save(self, file_path: str | Path) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def load(cls, file_path: str | Path) -> "Seq2SeqConfig":
        path = Path(file_path)
        with path.open("r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


class ConfigManager:
    _instance: "ConfigManager | None" = None
    _config: Seq2SeqConfig | None = None

    def __new__(cls) -> "ConfigManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._config is None:
            self._config = Seq2SeqConfig()

    @property
    def config(self) -> Seq2SeqConfig:
        if self._config is None:
            self._config = Seq2SeqConfig()
        return self._config

    def load(self, file_path: str | Path) -> None:
        self._config = Seq2SeqConfig.load(file_path)

    def save(self, file_path: str | Path) -> None:
        if self._config is None:
            raise ValueError("No configuration to save")
        self._config.save(file_path)

    def reset(self) -> None:
        self._config = Seq2SeqConfig()


# Global config instance
config_manager = ConfigManager()


def get_config() -> Seq2SeqConfig:
    return config_manager.config


def load_config(file_path: str | Path) -> None:
    config_manager.load(file_path)


def save_config(file_path: str | Path) -> None:
    config_manager.save(file_path)


if __name__ == "__main__":
    from dataclasses import asdict

    config = Seq2SeqConfig(
        model=ModelConfig(
            embedding_dim=256,
        ),
        optimizer=OptimizerConfig(
            learning_rate=5e-5,
        ),
        training=TrainingConfig(epochs=10),
    )

    print(config)
    print(asdict(config))
