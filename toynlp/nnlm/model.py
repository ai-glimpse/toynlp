import torch


class NNLM(torch.nn.Module):
    def __init__(
        self,
        seq_len: int = 6,
        vocab_size: int = 17964,
        embedding_dim: int = 100,
        hidden_dim: int = 60,
        with_direct_connection: bool = False,
        with_dropout: bool = True,
        dropout_rate: float = 0.2,
    ):
        """
        Args:
            seq_len: the length of the input sequence, the n in the paper
            vocab_size: vocabulary size, the |V| in the paper
            embedding_dim: embedding dimension, the m in the paper
            hidden_dim: hidden layer dimension, the h in the paper
            with_direct_connection: whether to use direct connection
            with_dropout: whether to use dropout
            dropout_rate: dropout rate
        """
        super(NNLM, self).__init__()
        self.with_direct_connection = with_direct_connection
        self.with_dropout = with_dropout
        # Embedding layer: |V| x m
        self.C = torch.nn.Embedding(vocab_size, embedding_dim)
        self.H = torch.nn.Linear(embedding_dim * (seq_len - 1), hidden_dim, bias=False)
        self.d = torch.nn.Parameter(torch.zeros(hidden_dim))
        self.U = torch.nn.Linear(hidden_dim, vocab_size, bias=False)
        self.activation = torch.nn.Tanh()

        self.b = torch.nn.Parameter(torch.zeros(vocab_size))
        self.W = torch.nn.Linear(embedding_dim * (seq_len - 1), vocab_size, bias=False)

        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (batch_size, seq_len-1) -> x: (batch_size, seq_len-1, embedding_dim)
        x = self.C(tokens)
        b, _, _ = x.shape
        # (batch_size, seq_len-1, embedding_dim) -> (batch_size, embedding_dim * (seq_len-1))
        x = x.reshape(b, -1)  # (batch_size, embedding_dim * (seq_len-1))
        if self.with_dropout:
            x = self.dropout(x)
        # (batch_size, embedding_dim * (seq_len-1)) -> (batch_size, vocab_size)
        x = self.b + self.U(self.activation(self.H(x) + self.d))  # no direct connection
        if self.with_direct_connection:
            x = x + self.W(x)
        # return logits
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # simple test
    n = 6
    vocab_size = 17964
    m = 100
    h = 60
    model = NNLM(seq_len=n, vocab_size=vocab_size, embedding_dim=m, hidden_dim=h)
    model.to(device)
    # |V |(1 + nm + h) + h(1 + (n − 1)m)
    print(
        sum(p.numel() for p in model.parameters()),
        vocab_size * (1 + n * m + h) + h * (1 + (n - 1) * m),
    )
    tokens = torch.randint(0, vocab_size, (2, 5)).to(device)
    print(model(tokens).shape)
