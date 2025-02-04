from pathlib import Path

from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer


class NNLMTokenizer:
    def __init__(self, model_path: str | None = None):
        self.tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        self._model_path = model_path

    @property
    def model_path(self) -> str:
        if self._model_path is not None:
            return self._model_path
        else:
            p = Path(__file__).parents[2] / "examples" / "nnlm" / "tokenizer-nnlm.json"
            p.parents[0].mkdir(parents=True, exist_ok=True)
            return str(p)

    def train(self, dataset: Dataset):
        trainer = WordLevelTrainer(
            vocab_size=20000, min_frequency=3, special_tokens=["[UNK]"]
        )
        self.tokenizer.train_from_iterator(dataset["text"], trainer=trainer)
        self.tokenizer.save(str(self.model_path))

    def load(self) -> Tokenizer:
        self.tokenizer = Tokenizer.from_file(self.model_path)
        return self.tokenizer


nnlm_tokenizer = NNLMTokenizer().load()


if __name__ == "__main__":
    # tokenizer = NNLMTokenizer()
    # dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    # tokenizer.train(dataset['train'])

    print(nnlm_tokenizer.encode("Hello World"))
    print(nnlm_tokenizer.decode([0, 1, 2, 3, 4, 5]))
