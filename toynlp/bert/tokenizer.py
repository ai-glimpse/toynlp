from datasets import Dataset, load_dataset
from tokenizers import Tokenizer, normalizers
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer, Lowercase
from tokenizers.pre_tokenizers import Punctuation, Sequence, Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer
from toynlp.bert.config import BertConfig, create_config_from_cli

from toynlp.paths import BERT_TOKENIZER_PATH


class BertTokenizer:
    def __init__(
        self,
    ) -> None:
        self.model_path = BERT_TOKENIZER_PATH
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        self.tokenizer = Tokenizer(WordPiece(vocab=None, unk_token="[UNK]", max_input_chars_per_word=20))
        self.tokenizer.pre_tokenizer = Sequence(
            [
                Punctuation(behavior="isolated"),
                Whitespace(),
            ],
        )
        self.tokenizer.normalizer = normalizers.Sequence(
            [
                BertNormalizer(),
                Lowercase(),
            ],  # type: ignore[assignment]
        )
        self.tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", 1),
                ("[SEP]", 2),
            ],
        )  # type: ignore[assignment]

    def train(self, dataset: Dataset, vocab_size: int, special_tokens: list[str]) -> Tokenizer:
        trainer = WordPieceTrainer(
            vocab_size=vocab_size,  # type: ignore[unknown-argument]
            special_tokens=special_tokens,  # type: ignore[unknown-argument]
        )
        self.tokenizer.train_from_iterator(dataset["text"], trainer=trainer)
        self.tokenizer.save(str(self.model_path))
        print(f"Tokenizer saved to {self.model_path}")
        return self.tokenizer

    def load(self) -> Tokenizer:
        self.tokenizer = Tokenizer.from_file(self.model_path.as_posix())
        return self.tokenizer


def train_tokenizer(config: BertConfig) -> None:
    """Train a tokenizer for the specified language using the provided config.

    Args:
        config: Configuration instance
    """
    tokenizer_path = BERT_TOKENIZER_PATH

    if not tokenizer_path.exists():
        bert_tokenizer = BertTokenizer()

        # Load dataset
        dataset = load_dataset(
            path=config.dataset_path,
            name=config.dataset_name,
            split=config.dataset_split_of_tokenizer,
        )

        # Prepare text data
        train_dataset = dataset.map(
            lambda batch: {"text": batch["text"]},
            remove_columns=["title"],
            batched=True,
            num_proc=config.num_proc,  # type: ignore[unknown-argument]
        )
        # Train tokenizer
        bert_tokenizer.train(
            train_dataset,  # type: ignore[unknown-argument]
            config.vocab_size,
            config.special_tokens,
        )
    else:
        print(f"Tokenizer already exists at {tokenizer_path}")


def test_tokenizers() -> None:
    """Test the trained tokenizers with sample texts."""
    print("\nTesting tokenizers:")
    tokenizer = BertTokenizer().load()

    # Test target language tokenizer
    text = "Two men are at the stove preparing food and vibecoding."
    output = tokenizer.encode(text)
    print(f"Text: {text}")
    print(f"Tokens: {output.tokens}")
    print(f"Token Ids: {output.ids}")
    print(f"Type Ids: {output.type_ids}")

    # encode tokens
    # token_ids = tokenizer.encode(output.tokens, is_pretokenized=True, add_special_tokens=False).ids
    token_ids = [tokenizer.token_to_id(token) for token in output.tokens]
    print(f"Token Ids(from tokens): {token_ids}")

    texts = ("Hello, y'all!", "How are you 😁 ?")
    output = tokenizer.encode(*texts)
    print(f"Texts: {texts}")
    print(f"Tokens: {output.tokens}")
    print(f"Type Ids: {output.type_ids}")


def main() -> None:
    """CLI entry point for training tokenizers using tyro config management."""
    # Load configuration from command line using tyro
    config = create_config_from_cli()

    print("Training tokenizers")
    print(f"Dataset: {config.dataset_path}")

    # Train tokenizers for both languages
    train_tokenizer(config)

    # Test tokenizers
    test_tokenizers()


if __name__ == "__main__":
    main()
