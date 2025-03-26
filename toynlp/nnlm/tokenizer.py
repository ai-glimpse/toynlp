from pathlib import Path

from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer


class NNLMTokenizer:
    def __init__(
        self,
        model_path: str | None = None,
        vocab_size: int = 20000,
    ) -> None:
        self._model_path = model_path
        self.vocab_size = vocab_size

        self.tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()

    @property
    def model_path(self) -> str:
        if self._model_path is not None:
            return self._model_path
        p = Path(__file__).parents[2] / "playground" / "nnlm" / "tokenizer.json"
        p.parents[0].mkdir(parents=True, exist_ok=True)
        return str(p)

    def train(self, dataset: Dataset) -> None:
        trainer = WordLevelTrainer(
            vocab_size=self.vocab_size,
            min_frequency=3,
            special_tokens=["[UNK]"],
        )
        self.tokenizer.train_from_iterator(dataset["text"], trainer=trainer)
        self.tokenizer.save(str(self.model_path))

    def load(self) -> Tokenizer:
        self.tokenizer = Tokenizer.from_file(self.model_path)
        return self.tokenizer


if __name__ == "__main__":
    from datasets import load_dataset

    tokenizer = NNLMTokenizer(vocab_size=20000)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenizer.train(dataset["train"])

    nnlm_tokenizer = NNLMTokenizer().load()
    print(nnlm_tokenizer.encode("Hello World"))
    print(nnlm_tokenizer.decode([0, 1, 2, 3, 4, 5]))
