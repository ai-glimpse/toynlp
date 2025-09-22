import random
import collections

from datasets import Dataset, load_dataset, DatasetDict
from tokenizers import Tokenizer
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from toynlp.bert.config import BertConfig
from toynlp.bert.tokenizer import BertTokenizer

bert_tokenizer = BertTokenizer().load()


# https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/tokenization.py#L78
def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    if isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    msg = f"Unsupported string type: {type(text)}"
    raise ValueError(msg)


# https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/create_pretraining_data.py#L418
def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])  # noqa: PYI024


def create_masked_lm_predictions(  # noqa: C901, PLR0912
    tokens,
    masked_lm_prob,
    max_predictions_per_seq,
    vocab_words,
    rng,
    do_whole_word_mask=False,
):
    """Creates the predictions for the masked LM objective."""
    cand_indexes = []  # type: ignore[var-annotated]
    for i, token in enumerate(tokens):
        if token in {"[CLS]", "[SEP]"}:
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if do_whole_word_mask and len(cand_indexes) >= 1 and token.startswith("##"):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(
        max_predictions_per_seq,
        max(1, round(len(tokens) * masked_lm_prob)),
    )

    masked_lms = []  # type: ignore[var-annotated]
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = "[MASK]"  # noqa: S105
            # 10% of the time, keep original
            elif rng.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def create_pretraining_examples_from_documents(  # noqa: C901, PLR0912, PLR0915
    random_document: list[list[str]],
    document: list[list[str]],
    max_seq_length: int,
    short_seq_prob: float,
    masked_lm_prob: float,
    max_predictions_per_seq: int,
    vocab_words: list[str],
    rng: random.Random,
) -> list[dict]:
    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    # for _ in range(10):
                    #     random_document_index = rng.randint(0, len(documents_dataset) - 1)
                    #     # if random_document_index != document_index:
                    #     if documents_dataset["document"][random_document_index][0] != document:
                    #         break

                    # random_document = documents_dataset["document"][random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append("[SEP]")
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions(
                    tokens,
                    masked_lm_prob,
                    max_predictions_per_seq,
                    vocab_words,
                    rng,
                )
                # instance = TrainingInstance(
                #     tokens=tokens,
                #     segment_ids=segment_ids,
                #     is_random_next=is_random_next,
                #     masked_lm_positions=masked_lm_positions,
                #     masked_lm_labels=masked_lm_labels,
                # )
                instance = {
                    "tokens": tokens,
                    "segment_ids": segment_ids,
                    "is_random_next": is_random_next,
                    "masked_lm_positions": masked_lm_positions,
                    "masked_lm_labels": masked_lm_labels,
                }
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances


def text_to_documents(text: str) -> list[list[str]]:
    all_documents = [[]]  # type: ignore[var-annotated]
    lines = text.split("\n")
    for line in lines:
        line = convert_to_unicode(line).strip()  # noqa: PLW2901
        # Empty lines are used as document delimiters
        if not line:
            all_documents.append([])
        # here we set add_special_tokens=False to avoid adding [CLS], [SEP] tokens
        # we will add them by hand later during pretraining data creation
        tokens = bert_tokenizer.encode(line, add_special_tokens=False).tokens
        if tokens:
            all_documents[-1].append(tokens)
    # Remove empty documents
    all_documents = [d for d in all_documents if d]
    random.shuffle(all_documents)
    return all_documents


def batch_text_to_documents(batch: list[str]) -> list[list[str]]:
    all_documents = []
    for text in batch:
        for doc in text_to_documents(text):
            all_documents.append(doc)  # noqa: PERF402
    return all_documents


def batch_create_pretraining_examples_from_documents(
    batch: list[list[list[str]]],
    max_seq_length: int,
    short_seq_prob: float,
    masked_lm_prob: float,
    max_predictions_per_seq: int,
    vocab_words: list[str],
    rng: random.Random,
) -> list[dict]:
    all_instances = []
    for i, document in enumerate(batch):
        # random doc except self
        random_document = batch[rng.choice([j for j in range(len(batch)) if j != i])]
        document_instances = create_pretraining_examples_from_documents(
            random_document,
            document,
            max_seq_length,
            short_seq_prob,
            masked_lm_prob,
            max_predictions_per_seq,
            vocab_words,
            rng,
        )
        all_instances.extend(document_instances)
    return all_instances


def get_dataset(
    dataset_path: str,
    dataset_name: str | None,
    split: str,
) -> Dataset:
    dataset = load_dataset(path=dataset_path, name=dataset_name, split=split, streaming=True)
    return dataset  # type: ignore[return-value]


def dataset_transform(raw_dataset: Dataset, config: BertConfig) -> Dataset:
    # Apply any necessary transformations to the dataset here
    # https://huggingface.co/docs/datasets/en/about_map_batch#input-size--output-size
    documents_dataset = raw_dataset.map(
        lambda batch: {"document": batch_text_to_documents(batch["text"])},
        batched=True,
        batch_size=8,
        # num_proc=12,
        remove_columns=["text", "title"],
    )
    pre_train_dataset = documents_dataset.map(
        lambda batch: {
            "tokens": [instance["tokens"] for instance in batch_instances],
            "segment_ids": [instance["segment_ids"] for instance in batch_instances],
            "is_random_next": [instance["is_random_next"] for instance in batch_instances],
            "masked_lm_positions": [instance["masked_lm_positions"] for instance in batch_instances],
            "masked_lm_labels": [instance["masked_lm_labels"] for instance in batch_instances],
        }
        if (
            batch_instances := batch_create_pretraining_examples_from_documents(
                batch["document"],
                max_seq_length=config.max_seq_length,
                short_seq_prob=config.short_seq_prob,
                masked_lm_prob=config.masked_lm_prob,
                max_predictions_per_seq=config.max_predictions_per_seq,
                vocab_words=list(bert_tokenizer.get_vocab().keys()),
                rng=random.Random(12345),
            )
        )
        else {},
        batched=True,
        batch_size=64,
        # num_proc=12,
        remove_columns=["document"],
    )

    return pre_train_dataset


def upload_pretrain_instance(all_pretrain_instances: Dataset):
    """Upload the pretraining dataset to Hugging Face dataset hub.

    Args:
        all_pretrain_instances (Dataset): The dataset to upload.
    """
    # Convert the dataset to a DatasetDict if not already
    dataset_dict = DatasetDict({"train": all_pretrain_instances})

    # Define the repository name on Hugging Face hub
    repo_name = "AI-Glimpse/bookcorpusopen-bert"

    # Push the dataset to the Hugging Face hub
    dataset_dict.push_to_hub(repo_name)

    print(f"Dataset successfully uploaded to Hugging Face hub under repository: {repo_name}")


def collate_fn(
    batch: list[dict],
    tokenizer: Tokenizer,
) -> dict[str, torch.Tensor]:
    batch_tokens = []
    batch_segment_ids = []
    batch_is_random_next = []
    batch_masked_lm_labels = []
    pad_id = tokenizer.token_to_id("[PAD]")

    batch_max_seq_length = max(len(item["tokens"]) for item in batch)

    for item in batch:
        batch_tokens.append(torch.tensor([tokenizer.token_to_id(token) for token in item["tokens"]]))
        batch_segment_ids.append(item["segment_ids"])
        batch_is_random_next.append(item["is_random_next"])
        # length batch_max_seq_length tensor for masked_lm_labels
        padded_masked_lm_label_token_ids = torch.full((batch_max_seq_length,), pad_id)
        for i, pos in enumerate(item["masked_lm_positions"].tolist()):
            padded_masked_lm_label_token_ids[pos] = tokenizer.token_to_id(item["masked_lm_labels"][i])
        batch_masked_lm_labels.append(padded_masked_lm_label_token_ids)
    batch_padded_token_id_tensor = pad_sequence(batch_tokens, padding_value=pad_id, batch_first=True)
    batch_segment_ids_tensor = pad_sequence(batch_segment_ids, padding_value=pad_id, batch_first=True)

    return {
        "tokens": batch_padded_token_id_tensor,
        "segment_ids": batch_segment_ids_tensor,
        "is_random_next": torch.tensor(batch_is_random_next, dtype=torch.long),
        "masked_lm_labels": torch.stack(batch_masked_lm_labels),
    }


def get_split_dataloader(
    dataset_path: str,
    split: str,
    config: BertConfig,
) -> DataLoader:
    # raw_dataset = get_dataset(dataset_path, None, split)  # type: ignore[call-arg]
    raw_dataset = load_dataset(path=dataset_path, name=None, split=split)
    raw_dataset = raw_dataset.shuffle(seed=42).to_iterable_dataset(num_shards=32)
    pretrain_dataset = dataset_transform(raw_dataset, config)
    dataloader = torch.utils.data.DataLoader(
        pretrain_dataset.with_format(type="torch"),
        batch_size=config.batch_size,
        collate_fn=lambda batch: collate_fn(batch, bert_tokenizer),
        num_workers=16,
        prefetch_factor=10,
        pin_memory=True,
        persistent_workers=True,
    )

    return dataloader


if __name__ == "__main__":
    from tqdm import tqdm

    config = BertConfig()

    # raw_dataset = get_dataset(config.dataset_path, config.dataset_name, split="train[:1790]")  # 10%
    # all_pretrain_instances = dataset_transform(raw_dataset)
    # print(f"Number of pre-training instances: {len(all_pretrain_instances)}")
    # upload_pretrain_instance(all_pretrain_instances)

    val_dataset_loader = get_split_dataloader(
        config.dataset_path,
        # config.dataset_split_of_model_val,
        "train[:10%]",
        config,
    )
    # print(f"Number of training batches: {len(val_dataset_loader)}")
    for i, batch in enumerate(tqdm(val_dataset_loader)):
        # Process each batch
        # print(batch)
        print(batch["tokens"].shape)
        print(batch["segment_ids"].shape)
        print(batch["is_random_next"].shape)
        print(batch["masked_lm_labels"].shape)
        print("=" * 20, i, "=" * 20)
