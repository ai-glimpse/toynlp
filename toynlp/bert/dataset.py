import random
import collections
from collections.abc import Iterator

from datasets import Dataset, load_dataset, DatasetDict, IterableDataset
from tokenizers import Tokenizer
import torch
from torch.utils.data import DataLoader, IterableDataset as TorchIterableDataset
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
    cand_indexes: list[list[int]] = []
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

    masked_lms: list[MaskedLmInstance] = []
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
    documents_dataset: Dataset,
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
                    for _ in range(10):
                        random_document_index = rng.randint(0, len(documents_dataset) - 1)
                        # if random_document_index != document_index:
                        if documents_dataset["document"][random_document_index][0] != document:
                            break

                    random_document = documents_dataset["document"][random_document_index]
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


def text_to_documents(text: str) -> list[list[list[str]]]:
    all_documents: list[list[list[str]]] = [[]]
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


def batch_text_to_documents(batch: list[str]) -> list[list[list[str]]]:
    all_documents = []
    for text in batch:
        for doc in text_to_documents(text):
            all_documents.append(doc)  # noqa: PERF402
    return all_documents


def batch_create_pretraining_examples_from_documents(
    documents_dataset: Dataset,
    batch: list[list[list[str]]],
    max_seq_length: int,
    short_seq_prob: float,
    masked_lm_prob: float,
    max_predictions_per_seq: int,
    vocab_words: list[str],
    rng: random.Random,
) -> list[dict]:
    all_instances = []
    for document in batch:
        document_instances = create_pretraining_examples_from_documents(
            documents_dataset,
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


class BufferedPretrainingDataset(TorchIterableDataset):
    """An IterableDataset that buffers pretraining instances to ensure consistent batch sizes."""

    def __init__(
        self,
        dataset: Dataset | IterableDataset,
        config: BertConfig,
        tokenizer: Tokenizer,
        documents_dataset: Dataset,
        vocab_words: list[str],
        buffer_size: int = 10000,
        seed: int = 12345,
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.tokenizer = tokenizer
        self.documents_dataset = documents_dataset
        self.vocab_words = vocab_words
        self.buffer_size = buffer_size
        self.seed = seed
        self.rng = random.Random(seed)

    def _generate_instances(self) -> Iterator[dict]:
        """Generate pretraining instances from the raw dataset."""
        buffer = []

        for item in self.dataset:
            # Convert text to documents
            if "text" in item:
                documents = text_to_documents(item["text"])

                # Create pretraining instances from documents
                for document in documents:
                    instances = create_pretraining_examples_from_documents(
                        self.documents_dataset,
                        document,  # Pass the document directly, not wrapped in a list
                        max_seq_length=self.config.max_seq_length,
                        short_seq_prob=self.config.short_seq_prob,
                        masked_lm_prob=self.config.masked_lm_prob,
                        max_predictions_per_seq=self.config.max_predictions_per_seq,
                        vocab_words=self.vocab_words,
                        rng=self.rng,
                    )
                    buffer.extend(instances)

                    # When buffer is full, shuffle and yield instances
                    while len(buffer) >= self.buffer_size:
                        # Shuffle buffer for better randomization
                        self.rng.shuffle(buffer)

                        # Yield instances from buffer
                        while len(buffer) > self.buffer_size // 2:  # Keep half for mixing
                            yield buffer.pop()

        # Yield remaining instances in buffer
        self.rng.shuffle(buffer)
        while buffer:
            yield buffer.pop()

    def __iter__(self) -> Iterator[dict]:
        """Return an iterator over the dataset."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process data loading
            return self._generate_instances()

        # Multi-process data loading
        # Each worker gets a different seed for randomization
        self.rng = random.Random(self.seed + worker_info.id)

        # If using multiple workers, we need to shard the dataset
        # This is a simplified approach - in production you might want more sophisticated sharding
        return self._generate_instances()


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


def collate_pretrain_instances(
    batch: list[dict],
    tokenizer: Tokenizer,
) -> dict[str, torch.Tensor]:
    """Collate function for pretraining instances that are already processed."""
    batch_tokens = []
    batch_segment_ids = []
    batch_is_random_next = []
    batch_masked_lm_labels = []
    pad_id = tokenizer.token_to_id("[PAD]")

    batch_max_seq_length = max(len(item["tokens"]) for item in batch)

    for item in batch:
        # Convert tokens to IDs
        token_ids = [tokenizer.token_to_id(token) for token in item["tokens"]]
        batch_tokens.append(torch.tensor(token_ids))

        # Handle segment_ids
        segment_ids = item["segment_ids"]
        if isinstance(segment_ids, list):
            batch_segment_ids.append(torch.tensor(segment_ids))
        else:
            batch_segment_ids.append(segment_ids)

        # Handle is_random_next
        batch_is_random_next.append(item["is_random_next"])

        # Create padded masked_lm_labels tensor
        padded_masked_lm_label_token_ids = torch.full((batch_max_seq_length,), pad_id)
        masked_positions = item["masked_lm_positions"]
        masked_labels = item["masked_lm_labels"]

        # Handle both list and tensor types for positions
        if torch.is_tensor(masked_positions):
            masked_positions = masked_positions.tolist()

        for i, pos in enumerate(masked_positions):
            padded_masked_lm_label_token_ids[pos] = tokenizer.token_to_id(masked_labels[i])

        batch_masked_lm_labels.append(padded_masked_lm_label_token_ids)

    # Pad sequences
    batch_padded_token_id_tensor = pad_sequence(batch_tokens, padding_value=pad_id, batch_first=True)
    batch_segment_ids_tensor = pad_sequence(batch_segment_ids, padding_value=pad_id, batch_first=True)

    return {
        "tokens": batch_padded_token_id_tensor,
        "segment_ids": batch_segment_ids_tensor,
        "is_random_next": torch.tensor(batch_is_random_next, dtype=torch.long),
        "masked_lm_labels": torch.stack(batch_masked_lm_labels),
    }


def dynamic_collate_fn(
    batch: list[dict],
    tokenizer: Tokenizer,
    config: BertConfig,
    documents_dataset: Dataset,
    vocab_words: list[str],
) -> dict[str, torch.Tensor]:
    """Collate function that performs transformations during training."""
    # Convert text to documents for each item in the batch
    batch_documents = []
    for item in batch:
        if "text" in item:
            documents = text_to_documents(item["text"])
            batch_documents.extend(documents)

    # Create pretraining examples from documents
    batch_instances = batch_create_pretraining_examples_from_documents(
        documents_dataset,
        batch_documents,
        max_seq_length=config.max_seq_length,
        short_seq_prob=config.short_seq_prob,
        masked_lm_prob=config.masked_lm_prob,
        max_predictions_per_seq=config.max_predictions_per_seq,
        vocab_words=vocab_words,
        rng=random.Random(12345),
    )

    if not batch_instances:
        # Return empty tensors if no instances were created
        return {
            "tokens": torch.empty((0, 0), dtype=torch.long),
            "segment_ids": torch.empty((0, 0), dtype=torch.long),
            "is_random_next": torch.empty((0,), dtype=torch.long),
            "masked_lm_labels": torch.empty((0, 0), dtype=torch.long),
        }

    # Process the instances into tensors
    batch_tokens = []
    batch_segment_ids = []
    batch_is_random_next = []
    batch_masked_lm_labels = []
    pad_id = tokenizer.token_to_id("[PAD]")

    batch_max_seq_length = max(len(instance["tokens"]) for instance in batch_instances)

    for instance in batch_instances:
        batch_tokens.append(torch.tensor([tokenizer.token_to_id(token) for token in instance["tokens"]]))
        batch_segment_ids.append(torch.tensor(instance["segment_ids"]))
        batch_is_random_next.append(instance["is_random_next"])

        # Create padded masked_lm_labels tensor
        padded_masked_lm_label_token_ids = torch.full((batch_max_seq_length,), pad_id)
        for i, pos in enumerate(instance["masked_lm_positions"]):
            padded_masked_lm_label_token_ids[pos] = tokenizer.token_to_id(instance["masked_lm_labels"][i])
        batch_masked_lm_labels.append(padded_masked_lm_label_token_ids)

    batch_padded_token_id_tensor = pad_sequence(batch_tokens, padding_value=pad_id, batch_first=True)
    batch_segment_ids_tensor = pad_sequence(batch_segment_ids, padding_value=pad_id, batch_first=True)

    return {
        "tokens": batch_padded_token_id_tensor,
        "segment_ids": batch_segment_ids_tensor,
        "is_random_next": torch.tensor(batch_is_random_next, dtype=torch.long),
        "masked_lm_labels": torch.stack(batch_masked_lm_labels),
    }


def get_dataset(
    dataset_path: str,
    dataset_name: str | None,
    split: str,
) -> Dataset:
    dataset = load_dataset(path=dataset_path, name=dataset_name, split=split)
    return dataset  # type: ignore[return-value]


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


def get_split_dataloader(
    dataset_path: str,
    split: str,
    config: BertConfig,
) -> DataLoader:
    """Get a DataLoader with consistent batch sizes using buffered streaming."""
    # Load the raw dataset
    raw_dataset = get_dataset(dataset_path, None, split)  # type: ignore[call-arg]

    # Create documents dataset for NSP task
    # We'll create a smaller sample for random document selection
    sample_size = min(1000, len(raw_dataset))  # type: ignore[arg-type]
    sample_dataset = raw_dataset.select(range(sample_size))  # type: ignore[attr-defined]

    documents_dataset = sample_dataset.map(
        lambda batch: {"document": batch_text_to_documents(batch["text"])},
        batched=True,
        batch_size=12,
        num_proc=1,  # Use single process for smaller sample
        remove_columns=["text", "title"],
    )

    # Create vocab_words list
    vocab_words = list(bert_tokenizer.get_vocab().keys())

    # Get seed from config or use default
    seed = getattr(config, "seed", 12345)

    # Create the buffered dataset
    buffered_dataset = BufferedPretrainingDataset(
        dataset=raw_dataset,  # type: ignore[arg-type]
        config=config,
        tokenizer=bert_tokenizer,
        documents_dataset=documents_dataset,
        vocab_words=vocab_words,
        buffer_size=10000,  # Adjust based on memory constraints
        seed=seed,
    )

    # Create DataLoader with the buffered dataset
    dataloader = DataLoader(
        buffered_dataset,
        batch_size=config.batch_size,
        collate_fn=lambda batch: collate_pretrain_instances(batch, bert_tokenizer),
        num_workers=0,  # Start with 0, can increase for parallel data loading
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch for consistent batch sizes
    )

    return dataloader


def get_split_dataloader_legacy(
    dataset_path: str,
    split: str,
    config: BertConfig,
) -> DataLoader:
    """Legacy dataloader that uses dynamic collate function (has batch size control issues)."""
    raw_dataset = get_dataset(dataset_path, None, split)  # type: ignore[call-arg]

    # Create documents dataset EXACTLY as in the original dataset_transform
    # This is needed for random document sampling in NSP task
    documents_dataset = raw_dataset.map(
        lambda batch: {"document": batch_text_to_documents(batch["text"])},
        batched=True,
        batch_size=12,
        num_proc=12,
        remove_columns=["text", "title"],
    )

    # Create vocab_words once to avoid recreating it on every batch
    vocab_words = list(bert_tokenizer.get_vocab().keys())

    dataloader = DataLoader(
        raw_dataset,  # type: ignore[arg-type]
        batch_size=config.batch_size,
        collate_fn=lambda batch: dynamic_collate_fn(batch, bert_tokenizer, config, documents_dataset, vocab_words),
    )

    return dataloader


if __name__ == "__main__":
    config = BertConfig()

    # Example usage with the new buffered streaming approach
    val_dataset_loader = get_split_dataloader(
        config.dataset_path,
        "train[:10]",
        config,
    )
    print("DataLoader created successfully (streaming mode - no fixed length)")

    # Test batch size consistency
    batch_sizes = []
    print("Testing batch consistency...")
    for i, batch in enumerate(val_dataset_loader):
        batch_size = batch["tokens"].shape[0]
        batch_sizes.append(batch_size)
        print(f"Batch {i + 1}: size={batch_size}")
        print(f"  tokens shape: {batch['tokens'].shape}")
        print(f"  segment_ids shape: {batch['segment_ids'].shape}")
        print(f"  is_random_next shape: {batch['is_random_next'].shape}")
        print(f"  masked_lm_labels shape: {batch['masked_lm_labels'].shape}")
        if i >= 4:  # Check first 5 batches
            break

    if batch_sizes:
        print("\nBatch size consistency check:")
        print(f"  Expected batch size: {config.batch_size}")
        print(f"  Actual batch sizes: {batch_sizes}")
        # For streaming datasets with drop_last=True, all batches should have the same size
        all_consistent = all(bs == config.batch_size for bs in batch_sizes)
        print(f"  All batches have consistent size: {all_consistent}")
        success_msg = "  SUCCESS: Batch size control is working!"
        issue_msg = "  ISSUE: Batch sizes are inconsistent"
        print(success_msg if all_consistent else issue_msg)
