import torch

from toynlp.gpt.config import GPTConfig
from toynlp.util import current_device


class PositionwiseFeedForward(torch.nn.Module):
    def __init__(self, d_model: int, d_feed_forward: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(d_model, d_feed_forward)
        self.linear2 = torch.nn.Linear(d_feed_forward, d_model)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(torch.nn.functional.gelu(self.linear1(x))))


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
            attention_weight = attention_weight.masked_fill(mask == 0, float("-inf"))

        attention_score = torch.nn.functional.softmax(attention_weight, dim=-1)

        # (b, h, s, s) @ (b, h, s, dv/h) -> (b, h, s, dv/h)
        value = attention_score @ v
        return value


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config: GPTConfig = config
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


class DecoderGPTBlock(torch.nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config: GPTConfig = config
        self.causal_mha = MultiHeadAttention(config=config)
        self.ffn = PositionwiseFeedForward(config.d_model, config.d_feed_forward, config.dropout_ratio)
        self.layernorm_causal_mha = torch.nn.LayerNorm(config.d_model)
        self.layernorm_ffn = torch.nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h1 = self.causal_mha(x, x, x, mask)
        y1 = self.layernorm_causal_mha(x + h1)

        h2 = self.ffn(y1)
        y2 = self.layernorm_ffn(y1 + h2)

        return y2


class Decoder(torch.nn.Module):
    def __init__(self, config: GPTConfig, padding_idx: int) -> None:
        super().__init__()
        self.config: GPTConfig = config
        self.padding_idx: int = padding_idx
        self.embedding = torch.nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=padding_idx,
        )
        # learnable positional embedding
        self.pe = torch.nn.Embedding(
            num_embeddings=config.max_seq_length,
            embedding_dim=config.d_model,
        )
        self.embedding_dropout = torch.nn.Dropout(p=config.dropout_ratio)
        self.layers = torch.nn.ModuleList(
            [DecoderGPTBlock(config) for _ in range(config.decoder_layers)],
        )
        self.lm_head = torch.nn.Linear(config.d_model, config.vocab_size, bias=False)
        # weight tying: The input embedding and output embedding share the same weights
        self.lm_head.weight = self.embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        embeddings = embeddings + self.pe(
            torch.arange(embeddings.size(1), device=embeddings.device).unsqueeze(0),
        )
        x = self.embedding_dropout(embeddings)

        for layer in self.layers:
            x = layer(x, mask)

        y = self.lm_head(x)
        return y


class GPTModel(torch.nn.Module):
    def __init__(self, config: GPTConfig, padding_idx: int) -> None:
        super().__init__()
        self.config: GPTConfig = config
        self.padding_idx: int = padding_idx
        self.decoder = Decoder(
            config=self.config,
            padding_idx=padding_idx,
        )
        self.device: torch.device = current_device

    def forward(self, input_token_ids: torch.Tensor) -> torch.Tensor:
        mask = self._get_mask(input_token_ids)
        decoder_output = self.decoder(input_token_ids, mask)
        return decoder_output

    def _get_mask(self, input_token_ids: torch.Tensor) -> torch.Tensor:
        # shape:  (batch_size, seq_length) -> (batch_size, 1, 1, seq_length)
        pad_mask = (input_token_ids != self.padding_idx).unsqueeze(1).unsqueeze(2)
        seq_length = input_token_ids.size(1)
        # shape: (seq_length, seq_length)
        causal_mask = torch.tril(torch.ones((seq_length, seq_length), device=self.device)).bool()
        # shape: (batch_size, 1, seq_length, seq_length)
        return pad_mask & causal_mask


if __name__ == "__main__":
    # test mha shapes
    config = GPTConfig()
    mha = MultiHeadAttention(config).to(device=current_device)
    x = torch.randn(2, 10, config.d_model, device=current_device)
    y = mha(x, x, x)
    print(y.shape)

    # test gpt model shapes
    input_token_ids = torch.randint(0, config.vocab_size, (2, 10), device=current_device)
    model = GPTModel(config, padding_idx=3).to(device=current_device)
    output = model(input_token_ids)
    print(output.shape)

    # model parameters: 116534784(in transformer)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params}")
    print(model)
