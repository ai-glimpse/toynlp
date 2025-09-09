import torch
from toynlp.bert.model import BertConfig, Bert, BertPretrainModel


def test_bert_architecture() -> None:
    # base bert config
    base_bert_config = BertConfig(
        max_seq_length=512,
        vocab_size=30522,
        d_model=768,
        attention_d_k=768,
        attention_d_v=768,
        head_num=12,
        d_feed_forward=3072,
        encoder_layers=12,
    )
    base_bert = Bert(base_bert_config, padding_idx=0)
    base_bert_param_count = sum(p.numel() for p in base_bert.parameters())
    # Make sure the parameter count matches the expected value for BERT base
    # 109482240 match EXACTLY with Huggingface implementation: BertModel.from_pretrained("bert-base-uncased")
    assert base_bert_param_count == 109482240  # 110M params

    # large bert config
    large_bert_config = BertConfig(
        max_seq_length=512,
        vocab_size=30522,
        d_model=1024,
        attention_d_k=1024,
        attention_d_v=1024,
        head_num=16,
        d_feed_forward=4096,
        encoder_layers=24,
    )
    large_bert_model = Bert(large_bert_config, padding_idx=0)
    large_bert_param_count = sum(p.numel() for p in large_bert_model.parameters())
    # Make sure the parameter count matches the expected value for BERT large
    # 335141888 match EXACTLY with Huggingface implementation: BertModel.from_pretrained("bert-large-uncased")
    assert large_bert_param_count == 335141888  # 335M params


def test_bert_model_shapes() -> None:
    """Test BERT base model forward pass and output shapes."""
    config = BertConfig()
    device = torch.device("cpu")

    # Test Bert shapes
    bert = Bert(config, padding_idx=0).to(device=device)
    input_tokens = torch.randint(0, config.vocab_size, (2, 10), dtype=torch.long, device=device)
    input_segments = torch.randint(0, 2, (2, 10), dtype=torch.long, device=device)
    z = bert(input_tokens, input_segments)
    assert z.shape == (2, 10, config.d_model)


def test_bert_pretrain_model_shapes() -> None:
    """Test BERT pretrain model forward pass and output shapes."""
    config = BertConfig()
    device = torch.device("cpu")

    # Test BertPretrainModel model shapes
    source_token_ids = torch.randint(0, config.vocab_size, (2, 10), device=device)
    source_segments = torch.randint(0, 2, (2, 10), device=device)
    model = BertPretrainModel(config, padding_idx=0).to(device=device)
    nsp_output, mlm_output = model(source_token_ids, source_segments)
    assert nsp_output.shape == (2, 2)  # batch_size=2, 2 classes for NSP
    assert mlm_output.shape == (2, 10, config.vocab_size)  # batch_size=2, seq_len=10, vocab_size
