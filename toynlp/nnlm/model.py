import torch

from toynlp.nnlm.config import NNLMConfig


class NNLM(torch.nn.Module):
    def __init__(
        self,
        config: NNLMConfig,
    ) -> None:
        """The Neural Network Language Model (NNLM) model.

        Args:
            config: NNLMConfig, the model configuration.
        """
        super().__init__()
        self.with_direct_connection = config.with_direct_connection
        self.with_dropout = config.with_dropout
        # Embedding layer: |V| x m
        self.C = torch.nn.Embedding(config.vocab_size, config.embedding_dim)
        self.H = torch.nn.Linear(
            config.embedding_dim * (config.context_size - 1),
            config.hidden_dim,
            bias=False,
        )
        self.d = torch.nn.Parameter(torch.zeros(config.hidden_dim))
        self.U = torch.nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.activation = torch.nn.Tanh()

        self.b = torch.nn.Parameter(torch.zeros(config.vocab_size))
        self.W = torch.nn.Linear(
            config.embedding_dim * (config.context_size - 1),
            config.vocab_size,
            bias=False,
        )

        self.dropout = torch.nn.Dropout(config.dropout_rate)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            tokens: torch.Tensor, (batch_size, seq_len-1), the input tokens.

        Returns:
            torch.Tensor, (batch_size, vocab_size), the logits.

        """
        # tokens: (batch_size, seq_len-1) -> x: (batch_size, seq_len-1, embedding_dim)
        x = self.C(tokens)
        b, _, _ = x.shape
        # (batch_size, seq_len-1, embedding_dim) -> (batch_size, embedding_dim * (seq_len-1))
        x = x.reshape(b, -1)  # (batch_size, embedding_dim * (seq_len-1))
        if self.with_dropout:
            x = self.dropout(x)
        # (batch_size, embedding_dim * (seq_len-1)) -> (batch_size, vocab_size)
        x1 = self.b + self.U(
            self.activation(self.H(x) + self.d),
        )  # no direct connection
        if not self.with_direct_connection:
            x = x1
        else:
            x2 = self.W(x)
            x = x1 + x2
        # return logits
        return x


if __name__ == "__main__":
    from toynlp.util import current_device

    config = NNLMConfig(
        context_size=6,
        vocab_size=20000,
        embedding_dim=100,
        hidden_dim=60,
        dropout_rate=0.2,
        with_dropout=True,
        with_direct_connection=True,
    )
    model = NNLM(config)
    model.to(current_device)
    n = config.context_size
    m = config.embedding_dim
    h = config.hidden_dim
    vocab_size = config.vocab_size
    print(
        sum(p.numel() for p in model.parameters()),
        # |V |(1 + nm + h) + h(1 + (n − 1)m)
        config.vocab_size * (1 + n * m + h) + h * (1 + (n - 1) * m),
    )
    tokens = torch.randint(0, vocab_size, (2, 5)).to(current_device)
    print(model(tokens).shape)
