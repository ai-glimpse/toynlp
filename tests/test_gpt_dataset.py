from types import SimpleNamespace
import torch

from toynlp.gpt.dataset import split_text_into_contexts


class DummyTokenizer:
    def __init__(self) -> None:
        self._vocab: dict[str, int] = {"<pad>": 0, "<eos>": 1}

    def encode(self, text: str) -> SimpleNamespace:
        ids = [self._vocab.setdefault(char, len(self._vocab)) for char in text]
        return SimpleNamespace(ids=ids)

    def token_to_id(self, token: str) -> int | None:
        return self._vocab.get(token)


def test_split_text_includes_eos_and_pads_last_chunk() -> None:
    tokenizer = DummyTokenizer()
    contexts = split_text_into_contexts(["abcd"], max_length=3, tokenizer=tokenizer)

    assert len(contexts) == 2
    expected_first = torch.tensor([2, 3, 4], dtype=torch.long)
    expected_second = torch.tensor([5, 1, 0], dtype=torch.long)
    assert torch.equal(contexts[0], expected_first)
    assert torch.equal(contexts[1], expected_second)


def test_split_text_inserts_single_eos_per_document() -> None:
    tokenizer = DummyTokenizer()
    texts = ["alpha", "<eos>should_be_literal"]
    contexts = split_text_into_contexts(texts, max_length=4, tokenizer=tokenizer)

    eos_id = tokenizer.token_to_id("<eos>")
    stacked = torch.stack(contexts)
    eos_count = int((stacked == eos_id).sum().item())
    assert eos_count == len(texts)
