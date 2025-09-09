import torch

from toynlp.bert.config import BertConfig
from toynlp.util import current_device


class BertEmbedding(torch.nn.Module):
    def __init__(self, vocab_size: int, max_seq_length: int, d_model: int, padding_idx: int, dropout: float) -> None:
        super().__init__()
        self.token_embedding = torch.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx,
        )
        self.position_embedding = torch.nn.Embedding(max_seq_length, d_model)
        # 2: sentence A, sentence B
        self.segment_embedding = torch.nn.Embedding(2, d_model)
        self.layer_norm = torch.nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.token_embedding(input_ids)
        position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        position_embeddings = self.position_embedding(position_ids)
        segment_embeddings = self.segment_embedding(segment_ids)
        embeddings = self.layer_norm(token_embeddings + position_embeddings + segment_embeddings)
        return self.dropout(embeddings)


class PositionwiseFeedForward(torch.nn.Module):
    def __init__(self, d_model: int, d_feed_forward: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(d_model, d_feed_forward)
        self.linear2 = torch.nn.Linear(d_feed_forward, d_model)
        self.dropout = torch.nn.Dropout(p=dropout)
        self.layer_norm = torch.nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.gelu(self.linear1(x))
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.layer_norm(x)
        return x


class ScaleDotProductionAttention(torch.nn.Module):
    def __init__(self, dropout_ratio: float = 0.1) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout_ratio)

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
        attention_score = self.dropout(attention_score)

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
        self.spda = ScaleDotProductionAttention(config.dropout_ratio)
        self.Wq = torch.nn.Linear(config.d_model, config.attention_d_k)
        self.Wk = torch.nn.Linear(config.d_model, config.attention_d_k)
        self.Wv = torch.nn.Linear(config.d_model, config.attention_d_v)
        self.Wo = torch.nn.Linear(config.attention_d_v, config.d_model)
        self.output_dropout = torch.nn.Dropout(p=config.dropout_ratio)
        self.output_layer_norm = torch.nn.LayerNorm(config.d_model, eps=1e-12)

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
        output = self.output_dropout(output)
        output = self.output_layer_norm(output)

        # (b, s, d_model)
        return output


class EncoderTransformerBlock(torch.nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.config = config
        self.mha = MultiHeadAttention(config)
        self.ffn = PositionwiseFeedForward(config.d_model, config.d_feed_forward, config.dropout_ratio)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h1 = self.mha(x, x, x, mask)
        y1 = x + h1

        h2 = self.ffn(y1)
        y2 = y1 + h2

        return y2


class Encoder(torch.nn.Module):
    def __init__(self, config: BertConfig, padding_idx: int) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = padding_idx
        self.layers = torch.nn.ModuleList(
            [EncoderTransformerBlock(config) for _ in range(config.encoder_layers)],
        )

    def forward(
        self,
        embedding: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = embedding
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Bert(torch.nn.Module):
    def __init__(self, config: BertConfig, padding_idx: int) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = padding_idx
        self.bert_embedding = BertEmbedding(
            vocab_size=config.vocab_size,
            max_seq_length=config.max_seq_length,
            d_model=config.d_model,
            padding_idx=padding_idx,
            dropout=config.dropout_ratio,
        )
        self.encoder = Encoder(
            config=self.config,
            padding_idx=padding_idx,
        )
        self.pooler = torch.nn.Linear(config.d_model, config.d_model)

    def forward(
        self,
        source_token_ids: torch.Tensor,
        source_segments: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_mask = self._get_source_mask(source_token_ids)
        embedding = self.bert_embedding(source_token_ids, source_segments)
        output = self.encoder(embedding, source_mask)
        output = self.pooler(output).tanh()
        return output

    def _get_source_mask(self, source_token_ids: torch.Tensor) -> torch.Tensor:
        # shape: (batch_size, 1, 1, source_seq_length)
        return (source_token_ids != self.padding_idx).unsqueeze(1).unsqueeze(2)


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
        self.layer_norm = torch.nn.LayerNorm(config.d_model, eps=1e-12)
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
        # x = self.linear(x)
        return x


class BertPretrainModel(torch.nn.Module):
    def __init__(self, config: BertConfig, padding_idx: int) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = padding_idx
        self.base_model = Bert(config, padding_idx)
        self.nsp_head = NSPHead(config)
        self.mlm_head = MLMHead(config, self.base_model.bert_embedding.token_embedding)
        self.device = current_device

    def forward(
        self,
        source_token_ids: torch.Tensor,
        source_segments: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_output = self.base_model(source_token_ids, source_segments)
        nsp_output = self.nsp_head(encoder_output)
        mlm_output = self.mlm_head(encoder_output)
        return nsp_output, mlm_output
