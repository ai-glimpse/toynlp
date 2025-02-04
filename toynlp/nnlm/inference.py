import torch

from toynlp.nnlm.tokenizer import nnlm_tokenizer


def evaluate_prompt(text: str):
    token_ids = nnlm_tokenizer.encode(text).ids
    token_ids_tensor = torch.tensor(token_ids).unsqueeze(0)
    model = torch.load("nnlm-ppl-159.pth", weights_only=False)
    model.eval()
    with torch.no_grad():
        logits = model(token_ids_tensor)
        pred = torch.argmax(logits, dim=1)
        print(nnlm_tokenizer.decode(pred.tolist()))


if __name__ == "__main__":
    evaluate_prompt("they both returned from previous")
