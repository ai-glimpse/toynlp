from pathlib import Path

from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer


class Word2VecTokenizer:
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
    from toynlp.word2vec.config import PathConfig
    from toynlp.word2vec.dataset import get_dataset

    tokenizer_path = PathConfig().tokenizer_path

    tokenizer = Word2VecTokenizer(model_path=tokenizer_path, vocab_size=20000)
    training_dataset = get_dataset()
    tokenizer.train(training_dataset["train"])  # type: ignore[unknown-argument]

    word2vec_tokenizer = Word2VecTokenizer(tokenizer_path).load()
    print(word2vec_tokenizer.encode("Hello World"))
    print(word2vec_tokenizer.decode([0, 1, 2, 3, 4, 5]))
