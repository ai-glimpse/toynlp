import torch

from toynlp.attention.config import AttentionConfig
from toynlp.attention.model import Attention, Decoder, Encoder, Seq2SeqAttentionModel


def test_encoder_architecture() -> None:
    """Test the encoder component of the attention model."""
    config = AttentionConfig(
        source_vocab_size=1000,
        target_vocab_size=800,
        embedding_dim=128,
        encoder_hidden_dim=256,
        decoder_hidden_dim=256,
        align_hidden_size=512,
        dropout_ratio=0.1,
    )

    encoder = Encoder(
        input_size=config.source_vocab_size,
        embedding_size=config.embedding_dim,
        encoder_hidden_dim=config.encoder_hidden_dim,
        decoder_hidden_dim=config.decoder_hidden_dim,
        dropout_ratio=config.dropout_ratio,
    )

    batch_size, seq_length = 4, 10
    input_tensor = torch.randint(0, config.source_vocab_size, (batch_size, seq_length))

    encoder_outputs, decoder_init_hidden = encoder(input_tensor)

    # Check output shapes
    assert encoder_outputs.shape == torch.Size([batch_size, seq_length, config.encoder_hidden_dim * 2])
    assert decoder_init_hidden.shape == torch.Size([1, batch_size, config.decoder_hidden_dim])


def test_attention_mechanism() -> None:
    """Test the attention mechanism component."""
    encoder_hidden_size = 256
    decoder_hidden_size = 256
    align_hidden_size = 512
    batch_size, source_seq_length = 4, 10

    attention = Attention(
        encoder_hidden_size=encoder_hidden_size,
        decoder_hidden_size=decoder_hidden_size,
        align_hidden_size=align_hidden_size,
    )

    # Create mock encoder outputs and decoder hidden state
    encoder_outputs = torch.randn(batch_size, source_seq_length, encoder_hidden_size * 2)
    decoder_hidden = torch.randn(1, batch_size, decoder_hidden_size)

    context, attention_weights = attention(encoder_outputs, decoder_hidden)

    # Check output shapes
    assert context.shape == torch.Size([batch_size, encoder_hidden_size * 2])
    assert attention_weights.shape == torch.Size([batch_size, source_seq_length])

    # Check that attention weights sum to 1
    assert torch.allclose(attention_weights.sum(dim=1), torch.ones(batch_size), atol=1e-6)


def test_decoder_architecture() -> None:
    """Test the decoder component of the attention model."""
    config = AttentionConfig(
        source_vocab_size=1000,
        target_vocab_size=800,
        embedding_dim=128,
        encoder_hidden_dim=256,
        decoder_hidden_dim=256,
        dropout_ratio=0.1,
    )

    decoder = Decoder(
        input_size=config.target_vocab_size,
        output_size=config.target_vocab_size,
        embedding_size=config.embedding_dim,
        encoder_hidden_dim=config.encoder_hidden_dim,
        decoder_hidden_dim=config.decoder_hidden_dim,
        dropout_ratio=config.dropout_ratio,
    )

    batch_size = 4
    input_ids = torch.randint(0, config.target_vocab_size, (batch_size, 1))
    context_vector = torch.randn(batch_size, config.encoder_hidden_dim * 2)
    hidden = torch.randn(1, batch_size, config.decoder_hidden_dim)

    output, new_hidden = decoder(input_ids, context_vector, hidden)

    # Check output shapes
    assert output.shape == torch.Size([batch_size, 1, config.target_vocab_size])
    assert new_hidden.shape == torch.Size([1, batch_size, config.decoder_hidden_dim])


def test_seq2seq_attention_model_forward() -> None:
    """Test the complete Seq2Seq attention model forward pass."""
    config = AttentionConfig(
        source_vocab_size=1000,
        target_vocab_size=800,
        embedding_dim=128,
        encoder_hidden_dim=256,
        decoder_hidden_dim=256,
        align_hidden_size=512,
        dropout_ratio=0.1,
        teacher_forcing_ratio=1.0,  # Use teacher forcing for deterministic testing
    )

    model = Seq2SeqAttentionModel(config)
    model.device = torch.device("cpu")  # type: ignore[unresolved-attribute]
    device = model.device

    batch_size, source_seq_length, target_seq_length = 4, 10, 8
    input_tensor = torch.randint(0, config.source_vocab_size, (batch_size, source_seq_length)).to(device)
    target_tensor = torch.randint(0, config.target_vocab_size, (batch_size, target_seq_length)).to(device)

    # Ensure target tensor starts with BOS token (index 1) for proper decoder initialization
    target_tensor[:, 0] = 1

    output = model(input_tensor, target_tensor)

    # Check output shape
    assert output.shape == torch.Size([batch_size, target_seq_length, config.target_vocab_size])

    # Check that outputs are finite (not NaN or infinite)
    assert torch.isfinite(output).all()

    # Check that the model produces some variation in outputs (not all zeros)
    assert not torch.allclose(output, torch.zeros_like(output))


def test_seq2seq_attention_model_parameter_count() -> None:
    """Test that the model has reasonable parameter count."""
    config = AttentionConfig(
        source_vocab_size=1000,
        target_vocab_size=800,
        embedding_dim=128,
        encoder_hidden_dim=256,
        decoder_hidden_dim=256,
        align_hidden_size=512,
        dropout_ratio=0.1,
    )

    model = Seq2SeqAttentionModel(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Check that we have a reasonable number of parameters
    assert total_params > 0
    assert trainable_params == total_params  # All parameters should be trainable by default

    # Basic sanity check - should have at least embedding parameters
    min_expected_params = (
        config.source_vocab_size * config.embedding_dim + config.target_vocab_size * config.embedding_dim
    )
    assert total_params > min_expected_params


def test_seq2seq_attention_model_gradient_flow() -> None:
    """Test that gradients flow properly through the model."""
    config = AttentionConfig(
        source_vocab_size=100,
        target_vocab_size=80,
        embedding_dim=32,
        encoder_hidden_dim=64,
        decoder_hidden_dim=64,
        align_hidden_size=128,
        dropout_ratio=0.1,
        teacher_forcing_ratio=1.0,
    )

    model = Seq2SeqAttentionModel(config)
    model.device = torch.device("cpu")  # type: ignore[unresolved-attribute]
    device = model.device

    batch_size, source_seq_length, target_seq_length = 2, 5, 4
    input_tensor = torch.randint(0, config.source_vocab_size, (batch_size, source_seq_length)).to(device)
    target_tensor = torch.randint(0, config.target_vocab_size, (batch_size, target_seq_length)).to(device)
    target_tensor[:, 0] = 1  # BOS token

    # Forward pass
    output = model(input_tensor, target_tensor)

    # Create a dummy loss
    loss = output.sum()

    # Backward pass
    loss.backward()

    # Check that gradients exist for all parameters
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient found for parameter: {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient found for parameter: {name}"
