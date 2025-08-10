from dataclasses import dataclass, asdict
from typing import Literal, Any
import tyro


@dataclass
class Word2VecConfig:
    """All in one for everything, without nested dataclasses."""
    
    # model selection
    model_name: Literal["cbow", "skip_gram"] = "cbow"
    
    # dataset configs
    dataset_path: str = "Salesforce/wikitext"
    dataset_name: str = "wikitext-103-raw-v1"
    
    # model configs  
    vocab_size: int = 20000
    embedding_dim: int = 256
    
    # data processing configs
    cbow_n_words: int = 4
    skip_gram_n_words: int = 4
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    
    # optimizer configs
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # training configs
    epochs: int = 10
    
    # wandb configs
    wandb_name: str | None = None
    wandb_project: str = "Word2Vec"

    def __post_init__(self) -> None:
        """Basic validation."""
        if self.epochs <= 0:
            raise ValueError("Epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

        if self.wandb_name is None:
            self.wandb_name = self._get_wandb_name()

    def _get_wandb_name(self) -> str:
        s = f"[{self.model_name}]embedding_dim:{self.embedding_dim}"
        return s
    
    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for logging/serialization."""
        return asdict(self)
    
    def get_model_config(self):
        """Get model config for backward compatibility."""
        return ModelConfig(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
        )
    
    def get_data_config(self):
        """Get data config for backward compatibility."""
        return DataConfig(
            cbow_n_words=self.cbow_n_words,
            skip_gram_n_words=self.skip_gram_n_words,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
        )


# Legacy config classes for backward compatibility
@dataclass
class ModelConfig:
    vocab_size: int
    embedding_dim: int


@dataclass  
class DataConfig:
    cbow_n_words: int
    skip_gram_n_words: int
    batch_size: int
    num_workers: int
    shuffle: bool

def create_config_from_cli() -> Word2VecConfig:
    """Create configuration from command line arguments using tyro."""
    return tyro.cli(Word2VecConfig)


if __name__ == "__main__":
    # Example of using tyro to parse command line arguments
    config = tyro.cli(Word2VecConfig)
    print("Configuration loaded:")
    print(f"  Model: {config.model_name}")
    print(f"  Dataset: {config.dataset_path}")
    print(f"  Embedding dimension: {config.embedding_dim}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.epochs}")
