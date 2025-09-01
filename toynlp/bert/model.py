import torch

from toynlp.bert.config import BertConfig
from toynlp.util import current_device


class BertEmbedding(torch.nn.Module):
    def __init__(self, vocab_size: int, max_length: int, d_model: int, padding_idx: int, dropout: float) -> None:
        super().__init__()
        self.token_embedding = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx,
        )
        # TODO: shall we sepcific padding_idx here?
        # self.position_embedding = torch.nn.Embedding(max_length, d_model, padding_idx=padding_idx)
        self.position_embedding = torch.nn.Embedding(max_length, d_model)
        # 2: sentence A, sentence B
        self.segment_embedding = torch.nn.Embedding(2, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.token_embedding(input_ids)
        position_ids = torch.arange(input_ids.size(1), device=current_device).unsqueeze(0)
        position_embeddings = self.position_embedding(position_ids)
        segment_embeddings = self.segment_embedding(segment_ids)
        embeddings = token_embeddings + position_embeddings + segment_embeddings
        return self.dropout(embeddings)


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
            # TODO: -inf?
            attention_weight = attention_weight.masked_fill(mask == 0, float("-10000"))

        attention_score = torch.nn.functional.softmax(attention_weight, dim=-1)

        # check nan
        # if torch.isnan(attention_score).any():
        #     print("[SelfAttention] NaN detected in attention_score")

        # (b, h, s, s) @ (b, h, s, dv/h) -> (b, h, s, dv/h)
        value = attention_score @ v
        return value


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, config: BertConfig) -> None:
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
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.config = config
        self.mha = MultiHeadAttention(config)
        self.ffn = PositionwiseFeedForward(config.d_model, config.d_feed_forward, config.dropout_ratio)
        self.layernorm_mha = torch.nn.LayerNorm(config.d_model)
        self.layernorm_ffn = torch.nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h1 = self.mha(x, x, x, mask)
        y1 = self.layernorm_mha(x + h1)

        h2 = self.ffn(y1)
        y2 = self.layernorm_ffn(y1 + h2)

        return y2


class Encoder(torch.nn.Module):
    def __init__(self, config: BertConfig, padding_idx: int) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = padding_idx
        self.bert_embedding = BertEmbedding(
            vocab_size=config.vocab_size,
            max_length=config.max_seq_length,
            d_model=config.d_model,
            padding_idx=padding_idx,
            dropout=config.dropout_ratio,
        )
        self.layers = torch.nn.ModuleList(
            [EncoderTransformerBlock(config) for _ in range(config.encoder_layers)],
        )

    def forward(
        self,
        source_token_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.bert_embedding(source_token_ids, segment_ids)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class NSPHead(torch.nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.pooler = torch.nn.Linear(self.d_model, self.d_model)
        self.fc = torch.nn.Linear(self.d_model, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Take the [CLS] token representation for classification
        cls_token = x[:, 0]
        return self.fc(self.pooler(cls_token).tanh())


class MLMHead(torch.nn.Module):
    def __init__(self, config: BertConfig, token_embedding: torch.nn.Embedding) -> None:
        super().__init__()
        self.dense = torch.nn.Linear(config.d_model, config.d_model)
        self.activation = torch.nn.GELU()
        self.layer_norm = torch.nn.LayerNorm(config.d_model)
        self.linear = torch.nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying: share weights with token embedding
        self.linear.weight = token_embedding.weight
        # Add bias parameter (not tied to embedding)
        self.bias = torch.nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense(x)
        x = self.activation(x)
        x = self.layer_norm(x)
        x = self.linear(x) + self.bias
        return x


class BertModel(torch.nn.Module):
    def __init__(self, config: BertConfig, padding_idx: int) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = padding_idx
        self.encoder = Encoder(
            config=self.config,
            padding_idx=padding_idx,
        )
        self.nsp_head = NSPHead(config)
        self.mlm_head = MLMHead(config, self.encoder.bert_embedding.token_embedding)
        self.device = current_device

    def forward(
        self,
        source_token_ids: torch.Tensor,
        source_segments: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_mask = self._get_source_mask(source_token_ids)
        encoder_output = self.encoder(source_token_ids, source_segments, source_mask)
        nsp_output = self.nsp_head(encoder_output)
        mlm_output = self.mlm_head(encoder_output)
        return nsp_output, mlm_output

    def _get_source_mask(self, source_token_ids: torch.Tensor) -> torch.Tensor:
        # shape: (batch_size, 1, 1, source_seq_length)
        return (source_token_ids != self.padding_idx).unsqueeze(1).unsqueeze(2)


if __name__ == "__main__":
    config = BertConfig()

    # test encoder shapes
    encoder = Encoder(config, padding_idx=0).to(device=current_device)
    input_tokens = torch.randint(0, config.vocab_size, (2, 10), device=current_device)
    input_segments = torch.randint(0, 2, (2, 10), device=current_device)
    z = encoder(input_tokens, input_segments)
    print(z.shape)  # (2, 10, d_model)

    # test transformer model shapes
    source_token_ids = torch.randint(0, config.vocab_size, (2, 10), device=current_device)
    source_segments = torch.randint(0, 2, (2, 10), device=current_device)
    model = BertModel(config, padding_idx=0).to(device=current_device)
    output = model(source_token_ids, source_segments)
    print(output[0].shape, output[1].shape)  # (2, 2), (2, 10, vocab_size)

    # BERTBASE (L=12, H=768, A=12, Total Parameters=110M)
    base_bert_config = BertConfig(
        vocab_size=30522,
        d_model=768,
        attention_d_k=768,
        attention_d_v=768,
        head_num=12,
        d_feed_forward=3072,
        encoder_layers=12,
    )
    base_bert_model = BertModel(base_bert_config, padding_idx=0).to(device=current_device)
    print("Base BERT model created:")
    print(f"  Layers: {base_bert_config.encoder_layers}")
    print(f"  Hidden size: {base_bert_config.d_model}")
    print(f"  Attention heads: {base_bert_config.head_num}")
    print(f"  Model parameters: {sum(p.numel() for p in base_bert_model.parameters()) / 1_000_000:.1f}M")

    # BERTLARGE (L=24, H=1024, A=16, Total Parameters=340M)
    large_bert_config = BertConfig(
        vocab_size=30522,
        d_model=1024,
        attention_d_k=1024,
        attention_d_v=1024,
        head_num=16,
        d_feed_forward=4096,
        encoder_layers=24,
    )
    large_bert_model = BertModel(large_bert_config, padding_idx=0).to(device=current_device)
    print("Large BERT model created:")
    print(f"  Layers: {large_bert_config.encoder_layers}")
    print(f"  Hidden size: {large_bert_config.d_model}")
    print(f"  Attention heads: {large_bert_config.head_num}")
    print(f"  Model parameters: {sum(p.numel() for p in large_bert_model.parameters()) / 1_000_000:.1f}M")
