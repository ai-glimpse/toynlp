import torch

from toynlp.attention.config import AttentionConfig
from toynlp.attention.model import Attention, Decoder, Encoder


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
