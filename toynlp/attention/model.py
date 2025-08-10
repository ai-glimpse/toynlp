import torch
import random
from toynlp.attention.config import AttentionConfig, create_config_from_cli
from toynlp.attention.tokenizer import AttentionTokenizer
from toynlp.util import current_device


class Encoder(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        embedding_size: int,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        dropout_ratio: float,
    ) -> None:
        super().__init__()
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.embedding = torch.nn.Embedding(
            num_embeddings=input_size,
            embedding_dim=embedding_size,
        )
        self.gru = torch.nn.GRU(
            input_size=embedding_size,
            hidden_size=encoder_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = torch.nn.Dropout(p=dropout_ratio)
        self.fc = torch.nn.Linear(
            in_features=encoder_hidden_dim * 2,  # Bidirectional GRU
            out_features=decoder_hidden_dim,  # Map to decoder hidden size
        )

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # (batch_size, seq_length) -> (batch_size, seq_length, embedding_size)
        embedded = self.dropout(self.embedding(input_ids))
        # output: (batch_size, seq_length, encoder_hidden_dim * 2)
        # hidden: (2 * num_layers, batch_size, encoder_hidden_dim)
        outputs, hidden = self.gru(embedded)
        # hidden cat: (batch_size, 2 * encoder_hidden_dim)
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        # decoder_init_hidden: (batch_size, decoder_hidden_dim)
        decoder_init_hidden = self.fc(hidden_cat).unsqueeze(0)
        return outputs, decoder_init_hidden


class Attention(torch.nn.Module):
    def __init__(
        self,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        align_hidden_size: int,
    ) -> None:
        super().__init__()
        self.Wa = torch.nn.Linear(decoder_hidden_size, align_hidden_size)
        self.Ua = torch.nn.Linear(encoder_hidden_size * 2, align_hidden_size)
        self.Va = torch.nn.Linear(align_hidden_size, 1)

    def forward(self, encoder_outputs: torch.Tensor, decoder_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention weights and context vector."""
        # encoder_outputs: (batch_size, source_seq_length, hidden_size * 2)  # Bidirectional GRU
        # decoder_hidden: (1, batch_size, decoder_hidden_size)

        # E matrix: e_ij = v_a^T * tanh(W_a * s_(i-1) + U_a * h_j)
        # hidden_state_merged: (batch_size, source_seq_length, align_hidden_size)
        hidden_state_merged = torch.tanh(
            self.Wa(decoder_hidden.permute(1, 0, 2)) + self.Ua(encoder_outputs),
        )
        # attention_value: (batch_size, source_seq_length, 1)
        attention_value = self.Va(hidden_state_merged)
        # attention_weight: (batch_size, source_seq_length, 1)
        attention_weight = torch.nn.functional.softmax(attention_value, dim=1)
        # Compute context vector: (batch_size, 1, hidden_size * 2)
        context = torch.bmm(attention_weight.transpose(1, 2), encoder_outputs)
        # Remove the sequence dimension: (batch_size, hidden_size * 2)
        context = context.squeeze(1)  # Remove the sequence dimension
        # Squeeze the last dimension of attention_weight: (batch_size, source_seq_length)
        attention_weight = attention_weight.squeeze(2)  # Remove the last dimension
        # Return context vector and attention weights
        return context, attention_weight


class Decoder(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        embedding_size: int,
        encoder_hidden_dim: int,
        decoder_hidden_dim: int,
        dropout_ratio: float,
    ) -> None:
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim

        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=input_size,
            embedding_dim=embedding_size,
        )
        self.gru = torch.nn.GRU(
            input_size=embedding_size + encoder_hidden_dim * 2,
            hidden_size=decoder_hidden_dim,
            batch_first=True,
        )
        self.fc = torch.nn.Linear(decoder_hidden_dim, output_size)
        self.dropout = torch.nn.Dropout(p=dropout_ratio)

    def forward(
        self,
        input_ids: torch.Tensor,
        context_vector: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decoder usually forwards one token at a time."""
        # (batch_size, seq_length) -> (batch_size, seq_length, embedding_size) -> (batch_size, embedding_size)
        target_embedded = self.dropout(self.embedding(input_ids)).squeeze(1)
        # concat context vector with target_embedded: (batch_size, embedding_size + encoder_hidden_dim * 2)
        gru_input = torch.cat((target_embedded, context_vector), dim=-1).unsqueeze(1)
        # output: (batch_size, seq_length, hidden_size)
        # hidden: (num_layers, batch_size, hidden_size)
        gru_output, hidden = self.gru(gru_input, hidden)
        # (batch_size, seq_length, hidden_size) -> (batch_size, seq_length, output_size)
        output = self.fc(gru_output)
        return output, hidden


class Seq2SeqAttentionModel(torch.nn.Module):
    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.force_teacher_ratio = self.config.teacher_forcing_ratio
        self.encoder = Encoder(
            input_size=config.source_vocab_size,
            embedding_size=config.embedding_dim,
            encoder_hidden_dim=config.encoder_hidden_dim,
            decoder_hidden_dim=config.decoder_hidden_dim,
            dropout_ratio=config.dropout_ratio,
        )
        self.attention = Attention(
            encoder_hidden_size=config.encoder_hidden_dim,
            decoder_hidden_size=config.decoder_hidden_dim,
            align_hidden_size=config.align_hidden_size,
        )
        self.decoder = Decoder(
            input_size=config.target_vocab_size,
            output_size=config.target_vocab_size,
            embedding_size=config.embedding_dim,
            encoder_hidden_dim=config.encoder_hidden_dim,
            decoder_hidden_dim=config.decoder_hidden_dim,
            dropout_ratio=config.dropout_ratio,
        )
        self.target_tokenizer = AttentionTokenizer(lang=self.config.target_lang).load()
        self.target_vocab_ids = list(self.target_tokenizer.get_vocab().values())
        self.device = current_device

    def forward(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        encoder_outputs, hidden = self.encoder(input_ids)
        batch_size, seq_length = target_ids.shape
        # Prepare the first input for the decoder, usually the start token
        # target_ids: (batch_size, squence_length)
        # decoder_input_ids: (batch_size, 1)
        decoder_input_ids = target_ids[:, 0].unsqueeze(1)  # Get the first token for the decoder
        outputs = torch.zeros(batch_size, seq_length, self.config.target_vocab_size).to(self.device)
        for t in range(1, seq_length):
            context, _ = self.attention(encoder_outputs, hidden)
            # decoder output: (batch_size, 1, target_vocab_size)
            decoder_output, hidden = self.decoder(decoder_input_ids, context, hidden)
            # Get the output for the current time step
            outputs[:, t, :] = decoder_output.squeeze(1)
            # (batch_size, target_vocab_size) -> (batch_size, 1)
            # Get the index of the highest probability token
            top_token_index = decoder_output.argmax(dim=-1).squeeze(1).tolist()
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < self.force_teacher_ratio
            if teacher_force:
                # Use the actual target token for the next input
                decoder_input_ids = target_ids[:, t].unsqueeze(1)
            else:
                # Use the predicted token for the next input
                # Convert token ids back to tensor
                token_ids = [self.target_vocab_ids[i] for i in top_token_index]
                decoder_input_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(1).to(self.device)
        return outputs


if __name__ == "__main__":
    config = create_config_from_cli()

    # test encoder
    # model = Seq2SeqAttentionModel(config)
    # input_tensor = torch.randint(0, config.source_vocab_size, (8, 10))
    # print(f"Input tensor shape: {input_tensor.shape}")
    # encoder = Encoder(
    #             input_size=config.source_vocab_size,
    #             embedding_size=config.embedding_dim,
    #             hidden_size=config.hidden_dim,
    #             dropout_ratio=config.dropout_ratio,
    #         )
    # encoder_output_hidden = encoder(input_tensor)
    # print(f"Encoder output hidden shape: {encoder_output_hidden.shape}")

    # test attention module
    # attention_alignment_model = Attention(hidden_size=config.hidden_dim, align_hidden_size=config.align_hidden_size)
    # print(attention_alignment_model)
    # encoder_outputs = torch.randn(8, 10, config.hidden_dim * 2)  # (batch_size, source_seq_length, hidden_size * 2)
    # decoder_hidden = torch.randn(8, config.hidden_dim)  # (batch_size, hidden_size)
    # context_vector = attention_alignment_model(encoder_outputs, decoder_hidden)
    # print(f"Context vector shape: {context_vector.shape}")  # Should be (batch_size, hidden_size * 2)

    # Example input
    model = Seq2SeqAttentionModel(config)
    model.to(current_device)
    input_tensor = torch.randint(0, config.source_vocab_size, (3, 10)).to(current_device)
    target_tensor = torch.randint(0, config.target_vocab_size, (3, 8)).to(current_device)
    print(f"Input tensor shape: {input_tensor.shape}, Target tensor shape: {target_tensor.shape}")

    output = model(input_tensor, target_tensor)
    print(f"Output tensor shape: {output.shape}")  # Should be (batch_size, seq_length, target_vocab_size)
