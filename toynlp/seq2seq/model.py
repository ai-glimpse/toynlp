import torch

from toynlp.seq2seq.config import ModelConfig

class Encoder(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        embedding_size: int,
        hidden_size: int,
        num_layers: int,
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

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(input_ids)
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

    def forward(self, input_ids: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Seq2SeqModel(torch.nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = torch.nn.Embedding(
            num_embeddings=config.input_vocab_size,
            embedding_dim=config.embedding_dim,
        )
        raise NotImplementedError
