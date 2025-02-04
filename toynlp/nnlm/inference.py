from pathlib import Path

import torch

from toynlp.device import current_device
from toynlp.nnlm.tokenizer import NNLMTokenizer


def evaluate_prompt(text: str):
    nnlm_tokenizer = NNLMTokenizer().load()
    token_ids = nnlm_tokenizer.encode(text).ids
    token_ids_tensor = torch.tensor(token_ids).unsqueeze(0).to(current_device)
    p = Path(__file__).parents[2] / "playground" / "nnlm" / "model.pth"
    model = torch.load(str(p), weights_only=False)
    model.eval()
    with torch.no_grad():
        logits = model(token_ids_tensor)
        pred = torch.argmax(logits, dim=1)
        print(nnlm_tokenizer.decode(pred.tolist()))


if __name__ == "__main__":
    evaluate_prompt("they both returned from previous")
