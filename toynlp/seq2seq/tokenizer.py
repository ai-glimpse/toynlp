from typing import Literal
from datasets import Dataset, load_dataset
from tokenizers import Tokenizer, normalizers
from tokenizers.models import WordLevel
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Punctuation, Sequence, Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer
from toynlp.seq2seq.config import Seq2SeqConfig

from toynlp.paths import SEQ2SEQ_TOKENIZER_EN_PATH, SEQ2SEQ_TOKENIZER_FR_PATH

class Seq2SeqTokenizer:
    def __init__(
        self,
        vocab_size: int = 80000,
        lang: Literal["en", "fr"] = "en",
    ) -> None:
        self.model_path = SEQ2SEQ_TOKENIZER_EN_PATH if lang == "en" else SEQ2SEQ_TOKENIZER_FR_PATH
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.vocab_size = vocab_size

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
                StripAccents(),
            ],  # type: ignore[assignment]
        )
        self.tokenizer.post_processor = TemplateProcessing(
            single="[BOS] $A [EOS]",
            special_tokens=[("[BOS]", 1), ("[EOS]", 2)],
        )  # type: ignore[assignment]

    def train(self, dataset: Dataset) -> Tokenizer:
        trainer = WordLevelTrainer(
            vocab_size=self.vocab_size,  # type: ignore[unknown-argument]
            min_frequency=50,  # type: ignore[unknown-argument]
            special_tokens=["[UNK]", "[BOS]", "[EOS]", "[PAD]"],  # type: ignore[unknown-argument]
        )
        self.tokenizer.train_from_iterator(dataset["text"], trainer=trainer)
        self.tokenizer.save(str(self.model_path))
        print(f"Tokenizer saved to {self.model_path}")
        return self.tokenizer

    def load(self) -> Tokenizer:
        self.tokenizer = Tokenizer.from_file(self.model_path.as_posix())
        return self.tokenizer


def train_tokenizer(config: Seq2SeqConfig, lang: Literal["en", "fr"] = "en") -> None:
    tokenizer_path = SEQ2SEQ_TOKENIZER_EN_PATH if lang == "en" else SEQ2SEQ_TOKENIZER_FR_PATH
    if not tokenizer_path.exists():
        seq2seq_tokenizer = Seq2SeqTokenizer(
            vocab_size=config.model.input_vocab_size,
            lang=lang,
        )
        dataset = load_dataset(path=config.dataset.path, name=config.dataset.name, split="train")
        lang_dataset = dataset.map(lambda batch: {"text": [item[lang] for item in batch["translation"]]}, 
                                   batched=True,
                                   remove_columns=["translation"],
                                   num_proc=8,  # type: ignore[call-arg]
                                   )
        seq2seq_tokenizer.train(dataset=lang_dataset)  # type: ignore[unknown-argument]
    else:
        print(f"Tokenizer already exists at {tokenizer_path}")


if __name__ == "__main__":
    train_tokenizer(Seq2SeqConfig(), lang="en")
    train_tokenizer(Seq2SeqConfig(), lang="fr")

    seq2seq_tokenizer_en = Seq2SeqTokenizer(lang="en").load()
    print(seq2seq_tokenizer_en.encode("Hello, World!").ids)
    print("|".join(seq2seq_tokenizer_en.decode([1, 19510, 4, 177, 681, 2]).split()))

    seq2seq_tokenizer_fr = Seq2SeqTokenizer(lang="fr").load()
    print(seq2seq_tokenizer_fr.encode("Bonjour le monde!").ids)
    print("|".join(seq2seq_tokenizer_fr.decode([1, 16891, 13, 381, 618, 2]).split()))

    print(seq2seq_tokenizer_en.token_to_id("[PAD]"))
    print(seq2seq_tokenizer_fr.token_to_id("[PAD]"))
