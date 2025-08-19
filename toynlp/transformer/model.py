import torch

from toynlp.transformer.config import TransformerConfig


class PositionalEncoding:
    def __init__(self, max_length: int, d_model: int) -> None:
        self.max_length = max_length
        self.d_model = d_model
        self.pe = self._pe_calculation()

    def _pe_calculation(self) -> torch.Tensor:
        position = torch.arange(self.max_length).unsqueeze(1)
        i = torch.arange(self.d_model).unsqueeze(0)
        angles = position * (1 / torch.pow(10000, (2 * (i // 2)) / self.d_model))

        pe = torch.zeros((self.max_length, self.d_model))
        pe[:, 0::2] = torch.sin(angles[:, 0::2])
        pe[:, 1::2] = torch.cos(angles[:, 1::2])
        return pe

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding to the input tensor."""
        # Here we assume the input tensor is of shape (batch_size, seq_length, d_model)
        return x + self.pe[: x.size(1), :]


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        # TODO: use only d_model
        self.q = torch.nn.Linear(config.d_model, config.attention_d_k)
        self.k = torch.nn.Linear(config.d_model, config.attention_d_k)
        self.v = torch.nn.Linear(config.d_model, config.attention_d_v)
        self.layernorm = torch.nn.LayerNorm(config.d_model)

        assert config.attention_d_k % config.head_num == 0
        assert config.attention_d_v % config.head_num == 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_length, d_model)
        batch_size, seq_length = x.size(0), x.size(1)
        # q, k shape: (batch_size, seq_length, attention_d_k)
        # v shape: (batch_size, seq_length, attention_d_v)
        q, k, v = self.q(x), self.k(x), self.v(x)

        q_k_head_dim = self.config.attention_d_k // self.config.head_num
        v_head_dim = self.config.attention_d_v // self.config.head_num
        # view as multi head
        q = q.view(batch_size, self.config.head_num, seq_length, q_k_head_dim)
        k = k.view(batch_size, self.config.head_num, seq_length, q_k_head_dim)
        v = v.view(batch_size, self.config.head_num, seq_length, v_head_dim)

        # (b, h, s, dK/h) @ (b, h, dK/h, s) -> (b, h, s, s)
        attention_weight = torch.nn.functional.softmax(q @ k.transpose(-2, -1) / (q_k_head_dim**0.5), dim=-1)
        # (b, h, s, s) @ (b, h, s, dv/h) -> (b, h, s, dv/h)
        attention = attention_weight @ v
        attention = attention.view(batch_size, seq_length, self.config.attention_d_v)

        # make sure: d_k = d_v = d_model/h
        assert q_k_head_dim * self.config.head_num == self.config.d_model
        assert v_head_dim * self.config.head_num == self.config.d_model

        # attention output: (b, h, s, dv/h)
        y = self.layernorm(x + attention)
        return y


class Encoder(torch.nn.Module):
    def __init__(self, config: TransformerConfig, padding_idx: int) -> None:
        super().__init__()
        self.config = config
        self.embedding = torch.nn.Embedding(
            num_embeddings=config.source_vocab_size,
            embedding_dim=config.d_model,
            padding_idx=padding_idx,
        )
        self.pe = PositionalEncoding(config.max_source_seq_length, config.d_model)
        self.embedding_dropout = torch.nn.Dropout(p=config.dropout_ratio)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        embeddings = self.pe.apply(embeddings)
        # we apply dropout to the sums of the embeddings and the positional encodings
        # in both the encoder and decoder stacks
        embeddings = self.embedding_dropout(embeddings)

        # TODO: more blocks
        return embeddings
