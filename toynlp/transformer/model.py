import torch
import random

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


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, config: TransformerConfig, masked: bool = False) -> None:
        super().__init__()
        self.config = config
        self.masked = masked
        self.Wq = torch.nn.Linear(config.d_model, config.attention_d_k)
        self.Wk = torch.nn.Linear(config.d_model, config.attention_d_k)
        self.Wv = torch.nn.Linear(config.d_model, config.attention_d_v)
        self.Wo = torch.nn.Linear(config.attention_d_v, config.d_model)
        self.dropout = torch.nn.Dropout(p=config.dropout_ratio)

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
        attention = self.dropout(attention)

        # TODO: Question：can we set dv != d_model?
        # make sure: d_k = d_v = d_model/h
        assert q_k_head_dim * self.config.head_num == self.config.d_model
        assert v_head_dim * self.config.head_num == self.config.d_model
        output = self.Wo(attention)  # (b, s, d_model)

        # (b, s, d_model)
        return output


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
        self.Wq = torch.nn.Linear(config.d_model, config.attention_d_k)
        self.Wk = torch.nn.Linear(config.d_model, config.attention_d_k)
        self.Wv = torch.nn.Linear(config.d_model, config.attention_d_v)
        self.Wo = torch.nn.Linear(config.attention_d_v, config.d_model)
        self.dropout = torch.nn.Dropout(p=config.dropout_ratio)

    def forward(self, encoder_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        q = self.Wq(x)
        # k and v are from the encoder output
        k = self.Wk(encoder_output)
        v = self.Wv(encoder_output)

        # x shape: (batch_size, seq_length, d_model)
        batch_size, target_seq_length = q.size(0), q.size(1)
        source_seq_length = k.size(1)

        q_k_head_dim = self.config.attention_d_k // self.config.head_num
        v_head_dim = self.config.attention_d_v // self.config.head_num
        # view as multi head
        q = q.view(
            batch_size,
            self.config.head_num,
            target_seq_length,
            self.config.attention_d_k // self.config.head_num,
        )
        k = k.view(batch_size, self.config.head_num, source_seq_length, q_k_head_dim)
        v = v.view(batch_size, self.config.head_num, source_seq_length, v_head_dim)

        # (b, h, target_seq_len, dK/h) @ (b, h, dK/h, source_seq_len) -> (b, h, target_seq_len, source_seq_len)
        attention_weight = q @ k.transpose(-2, -1) / (q_k_head_dim**0.5)
        attention_score = torch.nn.functional.softmax(attention_weight, dim=-1)
        # (b, h, target_seq_len, source_seq_len) @ (b, h, source_seq_len, dv/h) -> (b, h, target_seq_len, dv/h)
        attention = attention_score @ v
        # (b, h, target_seq_len, dv/h) -> (b, target_seq_len, h, dv/h)
        # TODO: Question: is the permute neccesary?
        attention = attention.permute(0, 2, 1, 3)
        # (b, target_seq_len, h, dv/h) -> (b, target_seq_len, dv)
        attention = attention.contiguous().view(batch_size, target_seq_length, self.config.attention_d_v)
        attention = self.dropout(attention)

        # TODO: Question：can we set dv != d_model?
        # make sure: d_k = d_v = d_model/h
        assert q_k_head_dim * self.config.head_num == self.config.d_model
        assert v_head_dim * self.config.head_num == self.config.d_model

        output = self.Wo(attention)

        # (b, s, d_model)
        return output


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
        h2 = self.cross_mha(q=y1, k=encoder_output, v=encoder_output)
        y2 = self.layernorm_cross_mha(y1 + h2)

        h3 = self.ffn(y2)
        y3 = self.layernorm_ffn(y2 + h3)

        return y3


class Decoder(torch.nn.Module):
    def __init__(self, config: TransformerConfig, padding_idx: int) -> None:
        super().__init__()
        self.config = config
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

    def forward(self, input_ids: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        embeddings = self.pe.apply(embeddings)
        x = self.embedding_dropout(embeddings)

        for layer in self.layers:
            x = layer(x, encoder_output)

        y = self.output_layer(x)
        return y


class TransformerModel(torch.nn.Module):
    def __init__(self, config: TransformerConfig, padding_idx: int) -> None:
        super().__init__()
        self.config = config
        self.force_teacher_ratio = self.config.teacher_forcing_ratio
        self.encoder = Encoder(
            config=self.config,
            padding_idx=padding_idx,
        )
        self.decoder = Decoder(
            config=self.config,
            padding_idx=padding_idx,
        )
        self.device = current_device

    def forward(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        encoder_memory = self.encoder(input_ids)
        batch_size, seq_length = target_ids.shape
        # Prepare the first input for the decoder, usually the start token
        # (batch_size, squence_length) -> (batch_size, 1)
        decoder_input_tensor = target_ids[:, 0].unsqueeze(1)  # Get the first token for the decoder
        outputs = torch.zeros(batch_size, seq_length, self.config.vocab_size).to(self.device)
        for t in range(seq_length):
            # decoder output: (batch_size, 1, target_vocab_size)
            decoder_output = self.decoder(decoder_input_tensor, encoder_memory)
            # Get the output for the current time step
            outputs[:, t, :] = decoder_output.squeeze(1)
            # (batch_size, target_vocab_size) -> (batch_size, 1)
            # Get the index of the highest probability token
            top_token_index = decoder_output.argmax(dim=-1).squeeze(1).tolist()
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < self.force_teacher_ratio
            if teacher_force:
                # Use the actual target token for the next input
                decoder_input_tensor = target_ids[:, t].unsqueeze(1)
            else:
                # Use the predicted token for the next input
                # Convert token ids back to tensor
                decoder_input_tensor = torch.tensor(top_token_index, dtype=torch.long, device=self.device).unsqueeze(1)
        return outputs


if __name__ == "__main__":
    # test mha shapes
    config = TransformerConfig()
    mha = MultiHeadSelfAttention(config).to(device=current_device)
    x = torch.randn(2, 10, config.d_model, device=current_device)
    y = mha(x)
    print(y.shape)

    # test masked mha shapes
    # mha_masked_config = TransformerConfig(d_model=6, attention_d_k=6, attention_d_v=6, head_num=3)
    # mha = MultiHeadSelfAttention(mha_masked_config, masked=True)
    # x = torch.randn(2, 5, 6)
    # y_masked = mha(x)
    # print(y_masked.shape)

    # test encoder shapes
    encoder = Encoder(config, padding_idx=0).to(device=current_device)
    x = torch.randint(0, config.vocab_size, (2, 10), device=current_device)
    z = encoder(x)
    print(z.shape)

    # test decoder shapes
    decoder = Decoder(config, padding_idx=0).to(device=current_device)
    y = decoder(x, z)
    print(y.shape)
