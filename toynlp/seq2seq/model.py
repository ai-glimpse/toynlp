import torch
import random
from toynlp.seq2seq.config import ModelConfig
from toynlp.seq2seq.tokenizer import Seq2SeqTokenizer
from toynlp.util import current_device


class Encoder(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        embedding_size: int,
        hidden_size: int,
        num_layers: int,
        dropout_ratio: float,
    ) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=input_size,
            embedding_dim=embedding_size,
        )
        self.lstm = torch.nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = torch.nn.Dropout(p=dropout_ratio)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # (batch_size, seq_length) -> (batch_size, seq_length, embedding_size)
        embedded = self.dropout(self.embedding(input_ids))
        # output: (batch_size, seq_length, hidden_size)
        # hidden: (num_layers, batch_size, hidden_size)
        # cell: (num_layers, batch_size, hidden_size)
        # we don't need the output, just the hidden and cell states
        _, (hidden, cell) = self.lstm(embedded)
        return hidden, cell


class Decoder(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        embedding_size: int,
        hidden_size: int,
        num_layers: int,
        dropout_ratio: float,
    ) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=input_size,
            embedding_dim=embedding_size,
        )
        self.lstm = torch.nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.dropout = torch.nn.Dropout(p=dropout_ratio)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decoder usually forwards one token at a time."""
        # (batch_size, seq_length) -> (batch_size, seq_length, embedding_size)
        target_embedded = self.dropout(self.embedding(input_ids))
        # output: (batch_size, seq_length, hidden_size)
        # hidden: (num_layers, batch_size, hidden_size)
        # cell: (num_layers, batch_size, hidden_size)
        lstm_output, (hidden, cell) = self.lstm(target_embedded, (hidden, cell))
        # (batch_size, seq_length, hidden_size) -> (batch_size, seq_length, output_size)
        output = self.fc(lstm_output)
        return output, hidden, cell


class Seq2SeqModel(torch.nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.force_teacher_ratio = self.config.teacher_forcing_ratio
        self.encoder = Encoder(
            input_size=config.source_vocab_size,
            embedding_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout_ratio=config.dropout_ratio,
        )
        self.decoder = Decoder(
            input_size=config.target_vocab_size,
            output_size=config.target_vocab_size,
            embedding_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout_ratio=config.dropout_ratio,
        )
        self.target_tokenizer = Seq2SeqTokenizer(lang=self.config.target_lang).load()
        self.target_vocab_ids = list(self.target_tokenizer.get_vocab().values())
        self.device = current_device

    def forward(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        hidden, cell = self.encoder(input_ids)
        batch_size, seq_length = target_ids.shape
        # Prepare the first input for the decoder, usually the start token
        # (batch_size, squence_length) -> (batch_size, 1)
        encoder_input_tensor = target_ids[:, 0].unsqueeze(1)  # Get the first token for the decoder
        outputs = torch.zeros(batch_size, seq_length, self.config.target_vocab_size).to(self.device)
        for t in range(seq_length):
            # decoder output: (batch_size, 1, target_vocab_size)
            decoder_output, hidden, cell = self.decoder(encoder_input_tensor, hidden, cell)
            # Get the output for the current time step
            outputs[:, t, :] = decoder_output.squeeze(1)
            # (batch_size, target_vocab_size) -> (batch_size, 1)
            # Get the index of the highest probability token
            top_token_index = decoder_output.argmax(dim=-1).squeeze(1).tolist()
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < self.force_teacher_ratio
            # teacher_force = True  # For simplicity, always use teacher forcing
            if teacher_force:
                # Use the actual target token for the next input
                encoder_input_tensor = target_ids[:, t].unsqueeze(1)
            else:
                # Use the predicted token for the next input
                # Convert token ids back to tensor
                token_ids = [self.target_vocab_ids[i] for i in top_token_index]
                encoder_input_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(1).to(self.device)
        return outputs


if __name__ == "__main__":
    from toynlp.seq2seq.config import get_config

    config = get_config()
    model = Seq2SeqModel(config.model)
    model.to(current_device)
    print(model)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Example input
    input_tensor = torch.randint(0, config.model.source_vocab_size, (2, 10)).to(current_device)
    target_tensor = torch.randint(0, config.model.target_vocab_size, (2, 8)).to(current_device)
    print(f"Input tensor shape: {input_tensor.shape}, Target tensor shape: {target_tensor.shape}")

    output = model(input_tensor, target_tensor)
    print(f"Output tensor shape: {output.shape}")  # Should be (batch_size, seq_length, target_vocab_size)
