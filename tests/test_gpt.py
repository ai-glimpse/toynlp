import torch
from toynlp.gpt.model import GPTConfig, GPTModel


def test_gpt_architecture() -> None:
    # base gpt config
    gpt_config = GPTConfig(
        max_seq_length=512,
        vocab_size=40478,
        d_model=768,
        attention_d_k=768,
        attention_d_v=768,
        head_num=12,
        d_feed_forward=3072,
        decoder_layers=12,
    )
    gpt = GPTModel(gpt_config, padding_idx=3)
    gpt_param_count = sum(p.numel() for p in gpt.parameters())
    # Make sure the parameter count matches the expected value for OpenAI GPT
    # 116534784 match EXACTLY with Huggingface: AutoModelForCausalLM.from_pretrained("openai-community/openai-gpt")
    assert gpt_param_count == 116534784  # 116M params


def test_gpt_model_shapes() -> None:
    """Test GPT model forward pass and output shapes."""
    config = GPTConfig()
    device = torch.device("cpu")

    # Test GPTModel shapes
    gpt_model = GPTModel(config, padding_idx=3).to(device=device)
    input_token_ids = torch.randint(0, config.vocab_size, (2, 10), dtype=torch.long, device=device)
    output = gpt_model(input_token_ids)
    assert output.shape == (2, 10, config.vocab_size)
