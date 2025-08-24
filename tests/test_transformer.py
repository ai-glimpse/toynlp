import torch
from toynlp.util import current_device
from toynlp.transformer.model import TransformerModel
from toynlp.transformer.config import TransformerConfig


def test_transformer_architecture():
    # Define a sample configuration
    config = TransformerConfig(
        vocab_size=100,
        d_model=32,
        attention_d_k=32,
        attention_d_v=32,
        head_num=4,
        d_feed_forward=64,
        dropout_ratio=0.1,
        encoder_layers=2,
        decoder_layers=2,
        max_source_seq_length=50,
        max_target_seq_length=50,
    )

    # Create sample input tensors
    batch_size = 2
    source_seq_length = 10
    target_seq_length = 15
    source_token_ids = torch.randint(0, config.vocab_size, (batch_size, source_seq_length), device=current_device)
    target_token_ids = torch.randint(0, config.vocab_size, (batch_size, target_seq_length), device=current_device)

    # Create the Transformer model
    model = TransformerModel(config, padding_idx=0)
    model.to(current_device)

    # Forward pass
    output = model(source_token_ids, target_token_ids)

    # Assertions
    assert output.shape == (batch_size, target_seq_length, config.vocab_size), "Output shape mismatch"
    assert not torch.isnan(output).any(), "Output contains NaN values"
