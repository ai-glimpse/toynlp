from pathlib import Path

from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer


class NNLMTokenizer:
    def __init__(
        self,
        model_path: Path,
        vocab_size: int = 20000,
    ) -> None:
        self.model_path = model_path
        self.vocab_size = vocab_size

        self.tokenizer = Tokenizer(WordLevel(vocab=None, unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()

    def train(self, dataset: Dataset) -> Tokenizer:
        trainer = WordLevelTrainer(
            vocab_size=self.vocab_size,  # type: ignore[unknown-argument]
            min_frequency=3,  # type: ignore[unknown-argument]
            special_tokens=["[UNK]"],  # type: ignore[unknown-argument]
        )
        self.tokenizer.train_from_iterator(dataset["text"], trainer=trainer)
        self.tokenizer.save(str(self.model_path))
        return self.tokenizer

    def load(self) -> Tokenizer:
        self.tokenizer = Tokenizer.from_file(self.model_path.as_posix())
        return self.tokenizer


if __name__ == "__main__":
    from datasets import load_dataset

    from toynlp.nnlm.config import NNLMPathConfig

    tokenizer_path = NNLMPathConfig().tokenizer_path

    tokenizer = NNLMTokenizer(model_path=tokenizer_path, vocab_size=20000)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenizer.train(dataset["train"])  # type: ignore[unknown-argument]

    nnlm_tokenizer = NNLMTokenizer(tokenizer_path).load()
    print(nnlm_tokenizer.encode("Hello World"))
    print(nnlm_tokenizer.decode([0, 1, 2, 3, 4, 5]))
