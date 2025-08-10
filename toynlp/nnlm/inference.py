import torch

from toynlp.util import current_device
from toynlp.nnlm.config import NNLMConfig
from toynlp.nnlm.tokenizer import NNLMTokenizer


def evaluate_prompt(text: str) -> None:
    config = NNLMConfig()
    tokenizer_model_path = config.tokenizer_path
    nnlm_model_path = config.model_path

    tokenizer = NNLMTokenizer(tokenizer_model_path).load()
    token_ids = tokenizer.encode(text).ids
    token_ids_tensor = torch.tensor(token_ids).unsqueeze(0).to(current_device)
    model = torch.load(str(nnlm_model_path), weights_only=False)
    model.eval()
    with torch.no_grad():
        logits = model(token_ids_tensor)
        pred = torch.argmax(logits, dim=1)
        print(tokenizer.decode(pred.tolist()))


if __name__ == "__main__":
    evaluate_prompt("they both returned from previous")
