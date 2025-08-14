import torch
from toynlp.fasttext.config import FastTextConfig


class FastTextModel(torch.nn.Module):
    def __init__(self, config: FastTextConfig) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embedding_dim)
        self.fc1 = torch.nn.Linear(config.embedding_dim, config.hidden_dim)
        self.fc2 = torch.nn.Linear(config.hidden_dim, config.num_classes)
        self.dropout = torch.nn.Dropout(p=config.dropout_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(x)).mean(dim=1)
        output = self.fc2(self.fc1(embedded))
        return output


if __name__ == "__main__":
    config = FastTextConfig(
        vocab_size=512,
        embedding_dim=300,
        hidden_dim=10,
        num_classes=5,
    )
    model = FastTextModel(config)
    print(model)
    # Create a dummy input tensor
    dummy_input = torch.randint(0, 512, (32, 20))  # Batch size of 32, sequence length of 20
    output = model(dummy_input)
    print(output.shape)  # Should be [32, 5]
