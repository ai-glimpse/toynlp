from datasets import Dataset, load_dataset
from tokenizers import Tokenizer, normalizers
from tokenizers.models import WordLevel
from tokenizers.normalizers import NFD, Lowercase
from tokenizers.pre_tokenizers import Punctuation, Sequence, Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer
from toynlp.seq2seq.config import Seq2SeqConfig

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


def train_tokenizer(lang: str, config: Seq2SeqConfig) -> None:
    """Train a tokenizer for the specified language using the provided config.

    Args:
        lang: Language code (e.g., 'en', 'de')
        config: Configuration object
    """
    tokenizer_path = SEQ2SEQ_TOKENIZER_PATH_MAP[lang]

    if not tokenizer_path.exists():
        seq2seq_tokenizer = Seq2SeqTokenizer(lang=lang)

        # Load dataset
        dataset = load_dataset(
            path=config.dataset_path,
            name=config.dataset_name,
            split="train",
        )

        # Prepare text data
        lang_dataset = dataset.map(
            lambda batch: {"text": list(batch[lang])},
            remove_columns=[config.source_lang, config.target_lang],
            batched=True,
            num_proc=config.num_proc,  # type: ignore[unknown-argument]
        )

        # Train tokenizer
        vocab_size = config.get_lang_vocab_size(lang)
        trainer = WordLevelTrainer(
            vocab_size=vocab_size,  # type: ignore[unknown-argument]
            min_frequency=config.min_frequency,  # type: ignore[unknown-argument]
            special_tokens=config.special_tokens,  # type: ignore[unknown-argument]
        )
        seq2seq_tokenizer.tokenizer.train_from_iterator(lang_dataset["text"], trainer=trainer)
        seq2seq_tokenizer.tokenizer.save(str(tokenizer_path))
        print(f"Tokenizer saved to {tokenizer_path}")
    else:
        print(f"Tokenizer already exists at {tokenizer_path}")


def train_all_tokenizers(config: Seq2SeqConfig) -> None:
    """Train tokenizers for both source and target languages."""
    train_tokenizer(config.source_lang, config)
    train_tokenizer(config.target_lang, config)


def test_tokenizers(config: Seq2SeqConfig) -> None:
    """Test the trained tokenizers with sample texts."""
    print("\nTesting tokenizers:")

    # Test source language tokenizer
    src_tokenizer = Seq2SeqTokenizer(lang=config.source_lang).load()
    src_text = "Zwei Männer stehen am Herd und bereiten Essen zu."
    src_tokens = src_tokenizer.encode(src_text).ids
    print(f"\n{config.source_lang.upper()}:")
    print(f"Text: {src_text}")
    print(f"Tokens: {src_tokens}")
    print(f"Decoded: {src_tokenizer.decode(src_tokens)}")

    # Test target language tokenizer
    tgt_tokenizer = Seq2SeqTokenizer(lang=config.target_lang).load()
    tgt_text = "Two men are at the stove preparing food."
    tgt_tokens = tgt_tokenizer.encode(tgt_text).ids
    print(f"\n{config.target_lang.upper()}:")
    print(f"Text: {tgt_text}")
    print(f"Tokens: {tgt_tokens}")
    print(f"Decoded: {tgt_tokenizer.decode(tgt_tokens)}")


def main() -> None:
    """CLI entry point for training tokenizers using tyro configuration."""
    # Load configuration from command line using tyro
    config = Seq2SeqConfig()

    print("=" * 60)
    print("SEQ2SEQ TOKENIZER TRAINING")
    print("=" * 60)
    print(f"Dataset: {config.dataset_path}")
    print(f"Languages: {config.source_lang}, {config.target_lang}")
    print("=" * 60)

    # Train tokenizers for both languages
    train_all_tokenizers(config)

    # Test tokenizers
    test_tokenizers(config)


if __name__ == "__main__":
    main()
