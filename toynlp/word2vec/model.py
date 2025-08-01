from torch import Tensor, nn

from toynlp.word2vec.config import ModelConfig


class Word2VecModel(nn.Module):
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


if __name__ == "__main__":
    import torch

    # Example usage
    config = ModelConfig(
        context_size=5,
        vocab_size=20000,
        embedding_dim=100,
    )
    model = Word2VecModel(config)
    example_input = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)  # Example input tensor
    output = model(example_input)
    print(output.shape)  # Should print torch.Size([1, 20000])

    # print model parameters count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
