from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn

from toynlp.word2vec.config import ModelConfig


class CbowModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
        )
        self.linear = nn.Linear(
            in_features=config.embedding_dim,
            out_features=config.vocab_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, context_size).

        Returns:
            Output tensor of shape (batch_size, vocab_size).
        """
        x = self.embedding(x)
        x = x.sum(dim=1)  # Sum the embeddings over the context size
        x = self.linear(x)
        return x


class SkipGramModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
        )
        self.linear = nn.Linear(
            in_features=config.embedding_dim,
            out_features=config.vocab_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, context_size).

        Returns:
            Output tensor of shape (batch_size, vocab_size).
        """
        x = self.embedding(x)
        x = self.linear(x)
        return x


if __name__ == "__main__":
    # Example usage
    config = ModelConfig(
        vocab_size=20000,
        embedding_dim=100,
    )
    cbow_model = CbowModel(config)
    # print model parameters count
    total_params = sum(p.numel() for p in cbow_model.parameters())
    print(f"Total parameters (CBOW): {total_params}")

    skip_gram_model = SkipGramModel(config)
    # print model parameters count
    total_params_skip_gram = sum(p.numel() for p in skip_gram_model.parameters())
    print(f"Total parameters (SkipGram): {total_params_skip_gram}")
