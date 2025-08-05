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
                # StripAccents(),
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


def train_tokenizer(config: Seq2SeqConfig, lang: str) -> None:
    tokenizer_path = SEQ2SEQ_TOKENIZER_PATH_MAP[lang]
    if not tokenizer_path.exists():
        seq2seq_tokenizer = Seq2SeqTokenizer(
            lang=lang,
        )
        dataset = load_dataset(path=config.dataset.path, name=config.dataset.name, split="train")
        # lang_dataset = dataset.map(
        #     lambda batch: {"text": [item[lang] for item in batch["translation"]]},
        #     batched=True,
        #     remove_columns=["translation"],
        #     num_proc=8,  # type: ignore[call-arg]
        # )
        lang_dataset = dataset.map(
            lambda batch: {"text": list(batch[lang])},
            remove_columns=["en", "de"],
            batched=True,
            num_proc=8,  # type: ignore[call-arg]
        )
        vocab_size = config.get_lang_vocab_size(lang)
        seq2seq_tokenizer.train(dataset=lang_dataset, vocab_size=vocab_size)  # type: ignore[unknown-argument]
    else:
        print(f"Tokenizer already exists at {tokenizer_path}")


if __name__ == "__main__":
    from toynlp.seq2seq.config import Seq2SeqConfig
    train_tokenizer(Seq2SeqConfig(), lang="en")
    train_tokenizer(Seq2SeqConfig(), lang="de")

    seq2seq_tokenizer_en = Seq2SeqTokenizer(lang="en").load()
    token_ids = seq2seq_tokenizer_en.encode("Two men are at the stove preparing food.").ids
    print(token_ids)
    print("|".join(seq2seq_tokenizer_en.decode(token_ids).split()))
    print(seq2seq_tokenizer_en.token_to_id("[PAD]"))

    seq2seq_tokenizer_de = Seq2SeqTokenizer(lang="de").load()
    token_ids = seq2seq_tokenizer_de.encode("Zwei Männer stehen am Herd und bereiten Essen zu.").ids
    print(token_ids)
    print("|".join(seq2seq_tokenizer_de.decode(token_ids).split()))
    print(seq2seq_tokenizer_de.token_to_id("[PAD]"))
