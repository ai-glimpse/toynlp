from datasets import Dataset, load_dataset
from tokenizers import Tokenizer, normalizers
from tokenizers.models import WordLevel
from tokenizers.normalizers import NFD, Lowercase
from tokenizers.pre_tokenizers import Punctuation, Sequence, Whitespace
from tokenizers.trainers import WordLevelTrainer
from toynlp.fasttext.config import FastTextConfig, create_config_from_cli

from toynlp.paths import FASTTEXT_TOKENIZER_PATH


class FastTextTokenizer:
    def __init__(
        self,
    ) -> None:
        self.model_path = FASTTEXT_TOKENIZER_PATH
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        self.tokenizer = Tokenizer(WordLevel(vocab=None, unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Sequence(
            [
                Punctuation(behavior="isolated"),
                Whitespace(),
            ],
        )
        self.tokenizer.normalizer = normalizers.Sequence(
            [
                NFD(),
                Lowercase(),
            ],  # type: ignore[assignment]
        )

    def train(self, dataset: Dataset, vocab_size: int) -> Tokenizer:
        trainer = WordLevelTrainer(
            vocab_size=vocab_size,  # type: ignore[unknown-argument]
            min_frequency=1,  # type: ignore[unknown-argument]
            special_tokens=["[UNK]", "[PAD]"],  # type: ignore[unknown-argument]
        )
        self.tokenizer.train_from_iterator(dataset["text"], trainer=trainer)
        self.tokenizer.save(str(self.model_path))
        print(f"Tokenizer saved to {self.model_path}")
        return self.tokenizer

    def load(self) -> Tokenizer:
        self.tokenizer = Tokenizer.from_file(self.model_path.as_posix())
        return self.tokenizer


def train_tokenizer(config: FastTextConfig) -> None:
    """Train a tokenizer for the specified language using the provided config.

    Args:
        config: Configuration instance
    """
    tokenizer_path = FASTTEXT_TOKENIZER_PATH

    if not tokenizer_path.exists():
        fasttext_tokenizer = FastTextTokenizer()

        # Load dataset
        dataset = load_dataset(
            path=config.dataset_path,
            name=config.dataset_name,
            split="train",
        )

        # Train tokenizer
        trainer = WordLevelTrainer(
            vocab_size=config.vocab_size,  # type: ignore[unknown-argument]
            min_frequency=config.min_frequency,  # type: ignore[unknown-argument]
            special_tokens=config.special_tokens,  # type: ignore[unknown-argument]
        )
        fasttext_tokenizer.tokenizer.train_from_iterator(dataset["text"], trainer=trainer)
        fasttext_tokenizer.tokenizer.save(str(tokenizer_path))
        print(f"Tokenizer saved to {tokenizer_path}")
    else:
        print(f"Tokenizer already exists at {tokenizer_path}")


def test_tokenizers() -> None:
    """Test the trained tokenizers with sample texts."""
    print("\nTesting tokenizers:")

    tokenizer = FastTextTokenizer().load()
    text = "If only to avoid making this type of film in the future."
    tokens = tokenizer.encode(text).ids
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {tokenizer.decode(tokens)}")


def main() -> None:
    """CLI entry point for training tokenizers using tyro config management."""
    config = create_config_from_cli()
    train_tokenizer(config)
    test_tokenizers()


if __name__ == "__main__":
    main()
