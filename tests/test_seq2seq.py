"""Tests for seq2seq module configuration and model."""

import pytest
import torch
from unittest.mock import Mock, patch
from toynlp.seq2seq.config import Seq2SeqConfig
from toynlp.seq2seq.model import Seq2SeqModel


def test_seq2seq_config():
    """Test that the new flattened config works correctly."""
    # Test default config
    config = Seq2SeqConfig()
    
    # Basic config checks
    assert config.dataset_path == "bentrevett/multi30k"
    assert config.source_lang == "de"
    assert config.target_lang == "en"
    assert config.embedding_dim == 256
    assert config.hidden_dim == 512
    assert config.num_layers == 2
    assert config.epochs == 10
    assert config.learning_rate == 1e-4
    
    # Test to_dict functionality
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert config_dict["source_lang"] == "de"
    assert config_dict["target_lang"] == "en"
    
    # Test vocab size functionality
    assert config.get_lang_vocab_size("de") == config.source_vocab_size
    assert config.get_lang_vocab_size("en") == config.target_vocab_size
    
    # Test error for unknown language
    with pytest.raises(ValueError):
        config.get_lang_vocab_size("fr")


@patch('toynlp.seq2seq.model.Seq2SeqTokenizer')
def test_seq2seq_model_with_config(mock_tokenizer_class):
    """Test that the seq2seq model works with the new config."""
    # Mock the tokenizer
    mock_tokenizer = Mock()
    mock_tokenizer.get_vocab.return_value = {f"token_{i}": i for i in range(100)}
    mock_tokenizer_class.return_value.load.return_value = mock_tokenizer
    
    config = Seq2SeqConfig(
        source_vocab_size=100,
        target_vocab_size=100,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=1,
    )
    
    model = Seq2SeqModel(config)
    
    # Check model attributes
    assert model.config.source_vocab_size == 100
    assert model.config.target_vocab_size == 100
    assert model.config.embedding_dim == 64
    assert model.config.hidden_dim == 128
    assert model.config.num_layers == 1
    
    # Test forward pass with small tensors
    batch_size = 2
    seq_len = 5
    
    input_ids = torch.randint(0, config.source_vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(0, config.target_vocab_size, (batch_size, seq_len))
    
    # This should not raise an error
    output = model(input_ids, target_ids)
    assert output.shape == (batch_size, seq_len, config.target_vocab_size)


def test_config_validation():
    """Test that config validation works."""
    # Test negative epochs
    with pytest.raises(ValueError, match="Epochs must be positive"):
        Seq2SeqConfig(epochs=-1)
    
    # Test negative learning rate
    with pytest.raises(ValueError, match="Learning rate must be positive"):
        Seq2SeqConfig(learning_rate=-0.1)