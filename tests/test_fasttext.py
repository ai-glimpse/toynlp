import torch

from toynlp.fasttext.config import FastTextConfig
from toynlp.fasttext.model import FastTextModel


def test_fasttext_model_architecture() -> None:
    """Test the FastText model architecture and forward pass."""
    config = FastTextConfig(
        vocab_size=1000,
        embedding_dim=100,
        hidden_dim=50,
        num_classes=5,
        dropout_ratio=0.1,
    )

    model = FastTextModel(config)

    # Test model parameters
    assert hasattr(model, "embedding")
    assert hasattr(model, "fc1")
    assert hasattr(model, "fc2")
    assert hasattr(model, "dropout")

    # Check embedding layer
    assert model.embedding.num_embeddings == config.vocab_size
    assert model.embedding.embedding_dim == config.embedding_dim

    # Check linear layers
    assert model.fc1.in_features == config.embedding_dim
    assert model.fc1.out_features == config.hidden_dim
    assert model.fc2.in_features == config.hidden_dim
    assert model.fc2.out_features == config.num_classes

    # Check dropout
    assert model.dropout.p == config.dropout_ratio


def test_fasttext_forward_pass() -> None:
    """Test the forward pass with different input shapes."""
    config = FastTextConfig(
        vocab_size=500,
        embedding_dim=64,
        hidden_dim=32,
        num_classes=3,
        dropout_ratio=0.2,
    )

    model = FastTextModel(config)
    model.eval()  # Set to evaluation mode to avoid randomness from dropout

    # Test with single sequence
    batch_size, seq_length = 1, 10
    input_tensor = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    output = model(input_tensor)

    assert output.shape == torch.Size([batch_size, config.num_classes])
    assert output.dtype == torch.float32

    # Test with batch of sequences
    batch_size, seq_length = 8, 15
    input_tensor = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    output = model(input_tensor)

    assert output.shape == torch.Size([batch_size, config.num_classes])

    # Test with different sequence lengths
    batch_size, seq_length = 4, 5
    input_tensor = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    output = model(input_tensor)

    assert output.shape == torch.Size([batch_size, config.num_classes])


def test_fasttext_gradient_flow() -> None:
    """Test that gradients flow properly through the model."""
    config = FastTextConfig(
        vocab_size=100,
        embedding_dim=32,
        hidden_dim=16,
        num_classes=2,
        dropout_ratio=0.0,  # No dropout for gradient test
    )

    model = FastTextModel(config)

    # Create input and target
    batch_size, seq_length = 2, 8
    input_tensor = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    target = torch.randint(0, config.num_classes, (batch_size,))

    # Forward pass
    output = model(input_tensor)
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(output, target)

    # Backward pass
    loss.backward()

    # Check that all parameters have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients"


def test_fasttext_embedding_averaging() -> None:
    """Test that the model correctly averages embeddings across sequence length."""
    config = FastTextConfig(
        vocab_size=10,
        embedding_dim=4,
        hidden_dim=2,
        num_classes=2,
        dropout_ratio=0.0,
    )

    model = FastTextModel(config)
    model.eval()

    # Create two identical sequences with different padding
    # First sequence: [1, 2, 3] (length 3)
    # Second sequence: [1, 2, 3, 0, 0] (length 5, with padding zeros)
    seq1 = torch.tensor([[1, 2, 3]])
    seq2 = torch.tensor([[1, 2, 3, 0, 0]])

    # Get embeddings manually to verify averaging behavior
    with torch.no_grad():
        # For seq1: should average 3 embeddings
        emb1 = model.embedding(seq1).mean(dim=1)  # [1, embedding_dim]

        # For seq2: should average 5 embeddings (including zeros)
        emb2 = model.embedding(seq2).mean(dim=1)  # [1, embedding_dim]

        # The averages should be different due to zero padding
        assert not torch.allclose(emb1, emb2, atol=1e-6)


def test_fasttext_output_consistency() -> None:
    """Test that the model produces consistent outputs for the same input."""
    config = FastTextConfig(
        vocab_size=50,
        embedding_dim=16,
        hidden_dim=8,
        num_classes=3,
        dropout_ratio=0.0,  # No dropout for consistency test
    )

    model = FastTextModel(config)
    model.eval()

    input_tensor = torch.randint(0, config.vocab_size, (1, 5))

    # Run the model multiple times
    with torch.no_grad():
        output1 = model(input_tensor)
        output2 = model(input_tensor)
        output3 = model(input_tensor)

    # Outputs should be identical
    assert torch.allclose(output1, output2, atol=1e-6)
    assert torch.allclose(output2, output3, atol=1e-6)


def test_fasttext_parameter_count() -> None:
    """Test that the model has the expected number of parameters."""
    config = FastTextConfig(
        vocab_size=1000,
        embedding_dim=100,
        hidden_dim=50,
        num_classes=5,
    )

    model = FastTextModel(config)

    # Calculate expected parameters
    # Embedding: vocab_size * embedding_dim
    # FC1: (embedding_dim + 1) * hidden_dim  # +1 for bias
    # FC2: (hidden_dim + 1) * num_classes    # +1 for bias
    expected_params = (
        config.vocab_size * config.embedding_dim +  # embedding
        (config.embedding_dim + 1) * config.hidden_dim +  # fc1
        (config.hidden_dim + 1) * config.num_classes  # fc2
    )

    actual_params = sum(p.numel() for p in model.parameters())
    assert actual_params == expected_params, f"Expected {expected_params} parameters, got {actual_params}"
