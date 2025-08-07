from datasets import Dataset, load_dataset
from tokenizers import Tokenizer, normalizers
from tokenizers.models import WordLevel
from tokenizers.normalizers import NFD, Lowercase
from tokenizers.pre_tokenizers import Punctuation, Sequence, Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer
from toynlp.seq2seq.config import get_config, load_config
import argparse

from toynlp.paths import SEQ2SEQ_TOKENIZER_PATH_MAP


class Seq2SeqTokenizer:
    def __init__(
        self,
        lang: str,
    ) -> None:
        self.model_path = SEQ2SEQ_TOKENIZER_PATH_MAP[lang]
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
        self.tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[("[BOS]", 1), ("[EOS]", 2)],
        )  # type: ignore[assignment]

    def train(self, dataset: Dataset, vocab_size: int) -> Tokenizer:
        trainer = WordLevelTrainer(
            vocab_size=vocab_size,  # type: ignore[unknown-argument]
            min_frequency=1,  # type: ignore[unknown-argument]
            special_tokens=["[UNK]", "[BOS]", "[EOS]", "[PAD]"],  # type: ignore[unknown-argument]
        )
        self.tokenizer.train_from_iterator(dataset["text"], trainer=trainer)
        self.tokenizer.save(str(self.model_path))
        print(f"Tokenizer saved to {self.model_path}")
        return self.tokenizer

    def load(self) -> Tokenizer:
        self.tokenizer = Tokenizer.from_file(self.model_path.as_posix())
        return self.tokenizer


def train_tokenizer(lang: str) -> None:
    """Train a tokenizer for the specified language using the global config.

    Args:
        lang: Language code (e.g., 'en', 'de')
    """
    config = get_config()
    tokenizer_path = SEQ2SEQ_TOKENIZER_PATH_MAP[lang]

    if not tokenizer_path.exists():
        seq2seq_tokenizer = Seq2SeqTokenizer(lang=lang)

        # Load dataset
        dataset = load_dataset(
            path=config.dataset.path,
            name=config.dataset.name,
            split="train",
        )

        # Prepare text data
        lang_dataset = dataset.map(
            lambda batch: {"text": list(batch[lang])},
            remove_columns=[config.dataset.source_lang, config.dataset.target_lang],
            batched=True,
            num_proc=config.tokenizer.num_proc,  # type: ignore[unknown-argument]
        )

        # Train tokenizer
        vocab_size = config.get_lang_vocab_size(lang)
        trainer = WordLevelTrainer(
            vocab_size=vocab_size,  # type: ignore[unknown-argument]
            min_frequency=config.tokenizer.min_frequency,  # type: ignore[unknown-argument]
            special_tokens=config.tokenizer.special_tokens,  # type: ignore[unknown-argument]
        )
        seq2seq_tokenizer.tokenizer.train_from_iterator(lang_dataset["text"], trainer=trainer)
        seq2seq_tokenizer.tokenizer.save(str(tokenizer_path))
        print(f"Tokenizer saved to {tokenizer_path}")
    else:
        print(f"Tokenizer already exists at {tokenizer_path}")


def train_all_tokenizers() -> None:
    """Train tokenizers for both source and target languages."""
    config = get_config()
    train_tokenizer(config.dataset.source_lang)
    train_tokenizer(config.dataset.target_lang)


def test_tokenizers() -> None:
    """Test the trained tokenizers with sample texts."""
    config = get_config()
    print("\nTesting tokenizers:")

    # Test source language tokenizer
    src_tokenizer = Seq2SeqTokenizer(lang=config.dataset.source_lang).load()
    src_text = "Zwei Männer stehen am Herd und bereiten Essen zu."
    src_tokens = src_tokenizer.encode(src_text).ids
    print(f"\n{config.dataset.source_lang.upper()}:")
    print(f"Text: {src_text}")
    print(f"Tokens: {src_tokens}")
    print(f"Decoded: {src_tokenizer.decode(src_tokens)}")

    # Test target language tokenizer
    tgt_tokenizer = Seq2SeqTokenizer(lang=config.dataset.target_lang).load()
    tgt_text = "Two men are at the stove preparing food."
    tgt_tokens = tgt_tokenizer.encode(tgt_text).ids
    print(f"\n{config.dataset.target_lang.upper()}:")
    print(f"Text: {tgt_text}")
    print(f"Tokens: {tgt_tokens}")
    print(f"Decoded: {tgt_tokenizer.decode(tgt_tokens)}")


def main() -> None:
    """CLI entry point for training tokenizers."""
    parser = argparse.ArgumentParser(description="Train sequence-to-sequence tokenizers")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--lang",
        type=str,
        choices=["en", "de"],
        help="Train tokenizer for specific language only",
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Skip testing the trained tokenizers",
    )
    args = parser.parse_args()

    # Load configuration
    load_config(args.config)

    # Train tokenizers
    if args.lang:
        train_tokenizer(args.lang)
    else:
        train_all_tokenizers()

    # Test tokenizers
    if not args.no_test:
        test_tokenizers()


if __name__ == "__main__":
    main()
