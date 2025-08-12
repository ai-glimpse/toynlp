import torch


class FastTextModel(torch.nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.fc = torch.nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        pooled = torch.mean(embedded, dim=1)
        output = self.fc(pooled)
        return output


if __name__ == "__main__":
    model = FastTextModel(vocab_size=512, embedding_dim=512, num_classes=5)
    print(model)
    # Create a dummy input tensor
    dummy_input = torch.randint(0, 512, (32, 10))  # Batch size of 32, sequence length of 10
    output = model(dummy_input)
    print(output.shape)  # Should be [32, 5]
