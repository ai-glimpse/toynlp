from toynlp.bert.model import BertConfig, Bert


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
