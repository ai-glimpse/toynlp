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



class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, config: TransformerConfig, masked: bool = False) -> None:
        super().__init__()
        self.config = config
        self.masked = masked
        self.Wq = torch.nn.Linear(config.d_model, config.attention_d_k)
        self.Wk = torch.nn.Linear(config.d_model, config.attention_d_k)
        self.Wv = torch.nn.Linear(config.d_model, config.attention_d_v)

        assert config.attention_d_k % config.head_num == 0
        assert config.attention_d_v % config.head_num == 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_length, d_model)
        batch_size, seq_length = x.size(0), x.size(1)
        # q, k shape: (batch_size, seq_length, attention_d_k)
        # v shape: (batch_size, seq_length, attention_d_v)
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)

        q_k_head_dim = self.config.attention_d_k // self.config.head_num
        v_head_dim = self.config.attention_d_v // self.config.head_num
        # view as multi head
        q = q.view(batch_size, self.config.head_num, seq_length, q_k_head_dim)
        k = k.view(batch_size, self.config.head_num, seq_length, q_k_head_dim)
        v = v.view(batch_size, self.config.head_num, seq_length, v_head_dim)

        # (b, h, s, dK/h) @ (b, h, dK/h, s) -> (b, h, s, s)
        attention_weight = q @ k.transpose(-2, -1) / (q_k_head_dim**0.5)
        if self.masked:
            tril = torch.tril(torch.ones_like(attention_weight), diagonal=0)
            attention_weight = attention_weight.masked_fill(tril == 0, float("-inf"))

        attention_score = torch.nn.functional.softmax(attention_weight, dim=-1)
        # (b, h, s, s) @ (b, h, s, dv/h) -> (b, h, s, dv/h)
        attention = attention_score @ v
        # (b, h, s, dv/h) -> (b, s, h, dv/h)
        # TODO: Question: is the permute neccesary?
        attention = attention.permute(0, 2, 1, 3)
        # (b, s, h, dv/h) -> (b, s, dv)
        attention = attention.contiguous().view(batch_size, seq_length, self.config.attention_d_v)

        # TODO: Question：can we set dv != d_model?
        # make sure: d_k = d_v = d_model/h
        assert q_k_head_dim * self.config.head_num == self.config.d_model
        assert v_head_dim * self.config.head_num == self.config.d_model

        # (b, s, d_model)
        return attention


class EncoderTransformerBlock(torch.nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.mha = MultiHeadSelfAttention(config)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(config.d_model, config.d_feed_forward),
            torch.nn.ReLU(),
            torch.nn.Linear(config.d_feed_forward, config.d_model),
        )
        self.layernorm_mha = torch.nn.LayerNorm(config.d_model)
        self.layernorm_ffn = torch.nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.mha(x)
        y1 = self.layernorm_mha(x + h1)

        h2 = self.ffn(y1)
        y2 = self.layernorm_ffn(y1 + h2)

        return y2


class Encoder(torch.nn.Module):
    def __init__(self, config: TransformerConfig, padding_idx: int) -> None:
        super().__init__()
        self.config = config
        self.embedding = torch.nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=padding_idx,
        )
        self.pe = PositionalEncoding(config.max_source_seq_length, config.d_model)
        self.embedding_dropout = torch.nn.Dropout(p=config.dropout_ratio)
        self.layers = torch.nn.ModuleList(
            [EncoderTransformerBlock(config) for _ in range(config.encoder_layers)],
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        embeddings = self.pe.apply(embeddings)
        # we apply dropout to the sums of the embeddings and the positional encodings
        # in both the encoder and decoder stacks
        x = self.embedding_dropout(embeddings)

        for layer in self.layers:
            x = layer(x)

        return x


class MultiHeadCrossAttention(torch.nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_length, d_model)
        batch_size, seq_length = x.size(0), x.size(1)

        q_k_head_dim = self.config.attention_d_k // self.config.head_num
        v_head_dim = self.config.attention_d_v // self.config.head_num
        # view as multi head
        q = q.view(batch_size, self.config.head_num, seq_length, q_k_head_dim)
        k = k.view(batch_size, self.config.head_num, seq_length, q_k_head_dim)
        v = v.view(batch_size, self.config.head_num, seq_length, v_head_dim)

        # (b, h, s, dK/h) @ (b, h, dK/h, s) -> (b, h, s, s)
        attention_weight = q @ k.transpose(-2, -1) / (q_k_head_dim**0.5)
        if self.masked:
            tril = torch.tril(torch.ones_like(attention_weight), diagonal=0)
            attention_weight = attention_weight.masked_fill(tril == 0, float("-inf"))

        attention_score = torch.nn.functional.softmax(attention_weight, dim=-1)
        # (b, h, s, s) @ (b, h, s, dv/h) -> (b, h, s, dv/h)
        attention = attention_score @ v
        # (b, h, s, dv/h) -> (b, s, h, dv/h)
        # TODO: Question: is the permute neccesary?
        attention = attention.permute(0, 2, 1, 3)
        # (b, s, h, dv/h) -> (b, s, dv)
        attention = attention.contiguous().view(batch_size, seq_length, self.config.attention_d_v)

        # TODO: Question：can we set dv != d_model?
        # make sure: d_k = d_v = d_model/h
        assert q_k_head_dim * self.config.head_num == self.config.d_model
        assert v_head_dim * self.config.head_num == self.config.d_model

        # (b, s, d_model)
        return attention


class DecoderTransformerBlock(torch.nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.masked_mha = MultiHeadSelfAttention(config=config, masked=True)
        self.cross_mha = MultiHeadCrossAttention(config)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(config.d_model, config.d_feed_forward),
            torch.nn.ReLU(),
            torch.nn.Linear(config.d_feed_forward, config.d_model),
        )
        self.layernorm_masked_mha = torch.nn.LayerNorm(config.d_model)
        self.layernorm_cross_mha = torch.nn.LayerNorm(config.d_model)
        self.layernorm_ffn = torch.nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        h1 = self.masked_mha(x)
        y1 = self.layernorm_masked_mha(x + h1)

        # q: y1(masked attention output), k: encoder output, v: encoder output
        h2 = self.cross_mha(y1, encoder_output, encoder_output)
        y2 = self.layernorm_cross_mha(y1 + h2)

        h3 = self.ffn(y2)
        y3 = self.layernorm_ffn(y2 + h3)

        return y3


class Decoder(torch.nn.Module):
    def __init__(self, config: TransformerConfig, padding_idx: int) -> None:
        super().__init__()
        self.config = config
        self.embedding = torch.nn.Embedding(
            num_embeddings=config.target_vocab_size,
            embedding_dim=config.d_model,
            padding_idx=padding_idx,
        )
        self.pe = PositionalEncoding(config.max_target_seq_length, config.d_model)
        self.embedding_dropout = torch.nn.Dropout(p=config.dropout_ratio)
        self.layers = torch.nn.ModuleList(
            [DecoderTransformerBlock(config) for _ in range(config.decoder_layers)],
        )
        self.output_layer = torch.nn.Linear(config.d_model, config.target_vocab_size)

    def forward(self, input_ids: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        embeddings = self.pe.apply(embeddings)
        x = self.embedding_dropout(embeddings)

        for layer in self.layers:
            x = layer(x, encoder_output)

        y = self.output_layer(x)
        return y


if __name__ == "__main__":
    # test mha shapes
    config = TransformerConfig()
    mha = MultiHeadSelfAttention(config)
    x = torch.randn(2, 10, config.d_model)
    y = mha(x)
    print(y.shape)

    # test masked mha shapes
    # mha_masked_config = TransformerConfig(d_model=6, attention_d_k=6, attention_d_v=6, head_num=3)
    # mha = MultiHeadSelfAttention(mha_masked_config, masked=True)
    # x = torch.randn(2, 5, 6)
    # y_masked = mha(x)
    # print(y_masked.shape)

    # test encoder shapes
    encoder = Encoder(config, padding_idx=0)
    x = torch.randint(0, config.vocab_size, (2, 10))
    z = encoder(x)
    print(z.shape)


    # test decoder shapes
    decoder = Decoder(config, padding_idx=0)
    y = decoder(x, z)
    print(y.shape)
