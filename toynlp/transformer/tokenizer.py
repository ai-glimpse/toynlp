from datasets import Dataset, load_dataset
from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.normalizers import NFD, Lowercase
from tokenizers.pre_tokenizers import Punctuation, Sequence, Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from toynlp.transformer.config import TransformerConfig, create_config_from_cli

from toynlp.paths import TRANSFORMER_TOKENIZER_PATH


class TransformerTokenizer:
    def __init__(
        self,
    ) -> None:
        self.model_path = TRANSFORMER_TOKENIZER_PATH
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        self.tokenizer = Tokenizer(BPE(vocab=None, unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Sequence(
            [
                Punctuation(behavior="merged_with_previous"),
                Whitespace(),
            ],
        )
        self.tokenizer.normalizer = normalizers.Sequence(
            [
                NFD(),
                # Lowercase(),
            ],  # type: ignore[assignment]
        )
        self.tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[("[BOS]", 1), ("[EOS]", 2)],
        )  # type: ignore[assignment]

    def train(self, dataset: Dataset, vocab_size: int) -> Tokenizer:
        trainer = BpeTrainer(
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


def train_tokenizer(config: TransformerConfig) -> None:
    """Train a tokenizer for the specified language using the provided config.

    Args:
        config: Configuration instance
    """
    tokenizer_path = TRANSFORMER_TOKENIZER_PATH

    if not tokenizer_path.exists():
        transformer_tokenizer = TransformerTokenizer()

        # Load dataset
        dataset = load_dataset(
            path=config.dataset_path,
            name=config.dataset_name,
            split="train",
        )

        # Prepare text data
        lang_dataset = dataset.map(
            lambda batch: {"text": list(batch[config.source_lang]) + list(batch[config.target_lang])},
            remove_columns=[config.source_lang, config.target_lang],
            batched=True,
            num_proc=config.num_proc,  # type: ignore[unknown-argument]
        )

        # Train tokenizer
        vocab_size = config.vocab_size
        trainer = BpeTrainer(
            vocab_size=vocab_size,  # type: ignore[unknown-argument]
            min_frequency=config.min_frequency,  # type: ignore[unknown-argument]
            special_tokens=config.special_tokens,  # type: ignore[unknown-argument]
        )
        transformer_tokenizer.tokenizer.train_from_iterator(lang_dataset["text"], trainer=trainer)
        transformer_tokenizer.tokenizer.save(str(tokenizer_path))
        print(f"Tokenizer saved to {tokenizer_path}")
    else:
        print(f"Tokenizer already exists at {tokenizer_path}")


def test_tokenizers(config: TransformerConfig) -> None:
    """Test the trained tokenizers with sample texts."""
    print("\nTesting tokenizers:")
    tokenizer = TransformerTokenizer().load()

    # Test source language tokenizer
    src_text = "Zwei Männer stehen am Herd und bereiten Essen zu."
    src_tokens = tokenizer.encode(src_text).ids
    print(f"\n{config.source_lang.upper()}:")
    print(f"Text: {src_text}")
    print(f"Tokens: {src_tokens}")
    print(f"Decoded: {tokenizer.decode(src_tokens)}")

    # Test target language tokenizer
    tgt_text = "Two men are at the stove preparing food."
    tgt_tokens = tokenizer.encode(tgt_text).ids
    print(f"\n{config.target_lang.upper()}:")
    print(f"Text: {tgt_text}")
    print(f"Tokens: {tgt_tokens}")
    print(f"Decoded: {tokenizer.decode(tgt_tokens)}")


def main() -> None:
    """CLI entry point for training tokenizers using tyro config management."""
    # Load configuration from command line using tyro
    config = create_config_from_cli()

    print(f"Training tokenizers for {config.source_lang} and {config.target_lang}")
    print(f"Dataset: {config.dataset_path}")

    # Train tokenizers for both languages
    train_tokenizer(config)

    # Test tokenizers
    test_tokenizers(config)


if __name__ == "__main__":
    main()
