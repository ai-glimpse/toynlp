import torch

from toynlp.device import current_device
from toynlp.nnlm.config import NNLMPathConfig
from toynlp.nnlm.tokenizer import NNLMTokenizer


def evaluate_prompt(text: str) -> None:
    path_config = NNLMPathConfig()
    tokenizer_model_path = path_config.tokenizer_path
    nnlm_model_path = path_config.model_path

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
