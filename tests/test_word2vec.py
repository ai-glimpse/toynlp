import torch

from toynlp.word2vec.config import Word2VecConfig
from toynlp.word2vec.model import CbowModel, SkipGramModel


def test_cbow_model_architecture() -> None:
    """Test CBOW model architecture and parameter count."""
    vocab_size = 1000
    embedding_dim = 100

    config = Word2VecConfig(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
    )

    model = CbowModel(config)

    # Parameter count: vocab_size * embedding_dim (embedding) + embedding_dim * vocab_size + vocab_size (linear)
    # = vocab_size * embedding_dim + embedding_dim * vocab_size + vocab_size
    # = 2 * vocab_size * embedding_dim + vocab_size
    expected_params = 2 * vocab_size * embedding_dim + vocab_size
    actual_params = sum(p.numel() for p in model.parameters())
    assert actual_params == expected_params

    # Test forward pass with context size 4
    batch_size = 2
    context_size = 4
    input_tensor = torch.randint(0, vocab_size, (batch_size, context_size))
    output = model(input_tensor)

    assert output.shape == torch.Size([batch_size, vocab_size])


def test_skip_gram_model_architecture() -> None:
    """Test Skip-gram model architecture and parameter count."""
    vocab_size = 1000
    embedding_dim = 100

    config = Word2VecConfig(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
    )

    model = SkipGramModel(config)

    # Parameter count: vocab_size * embedding_dim (embedding) + embedding_dim * vocab_size + vocab_size (linear)
    # = vocab_size * embedding_dim + embedding_dim * vocab_size + vocab_size
    # = 2 * vocab_size * embedding_dim + vocab_size
    expected_params = 2 * vocab_size * embedding_dim + vocab_size
    actual_params = sum(p.numel() for p in model.parameters())
    assert actual_params == expected_params

    # Test forward pass with single word input
    batch_size = 2
    input_tensor = torch.randint(0, vocab_size, (batch_size,))
    output = model(input_tensor)

    assert output.shape == torch.Size([batch_size, vocab_size])


def test_cbow_model_forward_pass() -> None:
    """Test CBOW model forward pass behavior."""
    vocab_size = 50
    embedding_dim = 10

    config = Word2VecConfig(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
    )

    model = CbowModel(config)
    model.eval()  # Set to evaluation mode

    # Test with different context sizes
    batch_size = 3
    context_size = 5
    input_tensor = torch.randint(0, vocab_size, (batch_size, context_size))

    with torch.no_grad():
        output = model(input_tensor)

    # Check output properties
    assert output.shape == torch.Size([batch_size, vocab_size])
    assert torch.isfinite(output).all()  # Check for NaN/Inf values

    # Test with different batch size
    different_batch_size = 1
    input_tensor_2 = torch.randint(0, vocab_size, (different_batch_size, context_size))

    with torch.no_grad():
        output_2 = model(input_tensor_2)

    assert output_2.shape == torch.Size([different_batch_size, vocab_size])


def test_skip_gram_model_forward_pass() -> None:
    """Test Skip-gram model forward pass behavior."""
    vocab_size = 50
    embedding_dim = 10

    config = Word2VecConfig(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
    )

    model = SkipGramModel(config)
    model.eval()  # Set to evaluation mode

    # Test with single word input
    batch_size = 3
    input_tensor = torch.randint(0, vocab_size, (batch_size,))

    with torch.no_grad():
        output = model(input_tensor)

    # Check output properties
    assert output.shape == torch.Size([batch_size, vocab_size])
    assert torch.isfinite(output).all()  # Check for NaN/Inf values

    # Test with different batch size
    different_batch_size = 1
    input_tensor_2 = torch.randint(0, vocab_size, (different_batch_size,))

    with torch.no_grad():
        output_2 = model(input_tensor_2)

    assert output_2.shape == torch.Size([different_batch_size, vocab_size])


def test_model_config() -> None:
    """Test model configuration."""
    config = Word2VecConfig(
        vocab_size=5000,
        embedding_dim=128,
    )

    assert config.vocab_size == 5000
    assert config.embedding_dim == 128

    # Test models can be created with config
    cbow_model = CbowModel(config)
    skip_gram_model = SkipGramModel(config)

    assert cbow_model.config == config
    assert skip_gram_model.config == config
