import torch

from toynlp.transformer.config import TransformerConfig
from toynlp.util import current_device


class PositionalEncoding:
    def __init__(self, max_length: int, d_model: int) -> None:
        self.max_length = max_length
        self.d_model = d_model
        self.pe = self._pe_calculation().to(device=current_device, dtype=torch.float32)

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


class PositionwiseFeedForward(torch.nn.Module):
    def __init__(self, d_model: int, d_feed_forward: int) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(d_model, d_feed_forward)
        self.linear2 = torch.nn.Linear(d_feed_forward, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(torch.nn.functional.relu(self.linear1(x)))


class ScaleDotProductionAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        head_dim = q.size(3)
        # (b, h, s, dK/h) @ (b, h, dK/h, s) -> (b, h, s, s)
        attention_weight = q @ k.transpose(-2, -1) / (head_dim**0.5)
        # pad mask
        if mask is not None:
            # TODO: -inf?
            attention_weight = attention_weight.masked_fill(mask == 0, float("-inf"))

        attention_score = torch.nn.functional.softmax(attention_weight, dim=-1)

        # check nan
        # if torch.isnan(attention_score).any():
        #     print("[SelfAttention] NaN detected in attention_score")

        # (b, h, s, s) @ (b, h, s, dv/h) -> (b, h, s, dv/h)
        value = attention_score @ v
        return value


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.spda = ScaleDotProductionAttention()
        self.Wq = torch.nn.Linear(config.d_model, config.attention_d_k)
        self.Wk = torch.nn.Linear(config.d_model, config.attention_d_k)
        self.Wv = torch.nn.Linear(config.d_model, config.attention_d_v)
        self.Wo = torch.nn.Linear(config.attention_d_v, config.d_model)
        self.dropout = torch.nn.Dropout(p=config.dropout_ratio)

        assert config.attention_d_k % config.head_num == 0
        assert config.attention_d_v % config.head_num == 0

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x shape: (batch_size, seq_length, d_model)
        batch_size = q.size(0)
        # q, k shape: (batch_size, seq_length, attention_d_k)
        # v shape: (batch_size, seq_length, attention_d_v)
        q, k, v = self.Wq(q), self.Wk(k), self.Wv(v)

        q_k_head_dim = self.config.attention_d_k // self.config.head_num
        v_head_dim = self.config.attention_d_v // self.config.head_num
        # view as multi head
        q = q.view(batch_size, q.size(1), self.config.head_num, q_k_head_dim).transpose(1, 2)
        k = k.view(batch_size, k.size(1), self.config.head_num, q_k_head_dim).transpose(1, 2)
        v = v.view(batch_size, v.size(1), self.config.head_num, v_head_dim).transpose(1, 2)

        spda_value = self.spda(q, k, v, mask)
        # (b, h, s, dv/h) -> (b, s, h, dv/h)
        spda_value = spda_value.transpose(1, 2)
        # (b, s, h, dv/h) -> (b, s, dv)
        spda_value = spda_value.contiguous().view(batch_size, spda_value.size(1), self.config.attention_d_v)
        output = self.Wo(spda_value)  # (b, s, d_model)

        # (b, s, d_model)
        return output


class EncoderTransformerBlock(torch.nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.mha = MultiHeadAttention(config)
        self.ffn = PositionwiseFeedForward(config.d_model, config.d_feed_forward)
        self.layernorm_mha = torch.nn.LayerNorm(config.d_model)
        self.layernorm_ffn = torch.nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h1 = self.mha(x, x, x, mask)
        y1 = self.layernorm_mha(x + h1)

        h2 = self.ffn(y1)
        y2 = self.layernorm_ffn(y1 + h2)

        return y2


class Encoder(torch.nn.Module):
    def __init__(self, config: TransformerConfig, padding_idx: int) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = padding_idx
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

    def forward(self, source_token_ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        embeddings = self.embedding(source_token_ids)
        embeddings = self.pe.apply(embeddings)
        # we apply dropout to the sums of the embeddings and the positional encodings
        # in both the encoder and decoder stacks
        x = self.embedding_dropout(embeddings)

        for layer in self.layers:
            x = layer(x, mask)

        return x


class DecoderTransformerBlock(torch.nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.causal_mha = MultiHeadAttention(config=config)
        self.cross_mha = MultiHeadAttention(config)
        self.ffn = PositionwiseFeedForward(config.d_model, config.d_feed_forward)
        self.layernorm_causal_mha = torch.nn.LayerNorm(config.d_model)
        self.layernorm_cross_mha = torch.nn.LayerNorm(config.d_model)
        self.layernorm_ffn = torch.nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        source_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h1 = self.causal_mha(x, x, x, target_mask)
        y1 = self.layernorm_causal_mha(x + h1)

        # q: y1(masked attention output), k: encoder output, v: encoder output
        h2 = self.cross_mha(y1, encoder_output, encoder_output, source_mask)
        y2 = self.layernorm_cross_mha(y1 + h2)

        h3 = self.ffn(y2)
        y3 = self.layernorm_ffn(y2 + h3)

        return y3


class Decoder(torch.nn.Module):
    def __init__(self, config: TransformerConfig, padding_idx: int) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = padding_idx
        self.embedding = torch.nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=padding_idx,
        )
        self.pe = PositionalEncoding(config.max_target_seq_length, config.d_model)
        self.embedding_dropout = torch.nn.Dropout(p=config.dropout_ratio)
        self.layers = torch.nn.ModuleList(
            [DecoderTransformerBlock(config) for _ in range(config.decoder_layers)],
        )
        self.output_layer = torch.nn.Linear(config.d_model, config.vocab_size)

    def forward(
        self,
        target_input_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        source_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = self.embedding(target_input_ids)
        embeddings = self.pe.apply(embeddings)
        x = self.embedding_dropout(embeddings)

        for layer in self.layers:
            x = layer(x, encoder_output, source_mask, target_mask)

        y = self.output_layer(x)
        return y


class TransformerModel(torch.nn.Module):
    def __init__(self, config: TransformerConfig, padding_idx: int) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = padding_idx
        self.encoder = Encoder(
            config=self.config,
            padding_idx=padding_idx,
        )
        self.decoder = Decoder(
            config=self.config,
            padding_idx=padding_idx,
        )
        self.device = current_device

    def forward(self, source_token_ids: torch.Tensor, target_token_ids: torch.Tensor) -> torch.Tensor:
        source_mask = self._get_source_mask(source_token_ids)
        target_mask = self._get_target_mask(target_token_ids)
        encoder_output = self.encoder(source_token_ids, source_mask)
        decoder_output = self.decoder(target_token_ids, encoder_output, source_mask, target_mask)
        return decoder_output

    def _get_source_mask(self, source_token_ids: torch.Tensor) -> torch.Tensor:
        # shape: (batch_size, 1, 1, source_seq_length)
        return (source_token_ids != self.padding_idx).unsqueeze(1).unsqueeze(2)

    def _get_target_mask(self, target_token_ids: torch.Tensor) -> torch.Tensor:
        # shape: (batch_size, 1, 1, target_seq_length)
        pad_mask = (target_token_ids != self.padding_idx).unsqueeze(1).unsqueeze(2)
        target_seq_length = target_token_ids.size(1)
        # shape: (batch_size, 1, target_seq_length, target_seq_length)
        causal_mask = torch.tril(torch.ones((target_seq_length, target_seq_length), device=self.device)).bool()

        return pad_mask & causal_mask


if __name__ == "__main__":
    # test mha shapes
    config = TransformerConfig()
    mha = MultiHeadAttention(config).to(device=current_device)
    x = torch.randn(2, 10, config.d_model, device=current_device)
    y = mha(x, x, x)
    print(y.shape)

    # test masked mha shapes
    # mha_masked_config = TransformerConfig(d_model=6, attention_d_k=6, attention_d_v=6, head_num=3)
    # mha = MultiHeadSelfAttention(mha_masked_config, masked=True)
    # x = torch.randn(2, 5, 6)
    # y_masked = mha(x)
    # print(y_masked.shape)

    # test encoder shapes
    encoder = Encoder(config, padding_idx=3).to(device=current_device)
    x = torch.randint(0, config.vocab_size, (2, 10), device=current_device)
    z = encoder(x)
    print(z.shape)

    # test decoder shapes
    decoder = Decoder(config, padding_idx=3).to(device=current_device)
    y = decoder(x, z)
    print(y.shape)

    # test transformer model shapes
    source_token_ids = torch.randint(0, config.vocab_size, (2, 10), device=current_device)
    target_token_ids = torch.randint(0, config.vocab_size, (2, 10), device=current_device)
    model = TransformerModel(config, padding_idx=3).to(device=current_device)
    output = model(source_token_ids, target_token_ids)
    print(output.shape)
