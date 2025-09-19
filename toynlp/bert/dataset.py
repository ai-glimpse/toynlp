import random
import collections
import time

from datasets import Dataset, load_dataset
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
    msg = f"Unsupported string type: {type(text)}"
    raise ValueError(msg)


# https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/create_pretraining_data.py#L418


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


def get_split_dataloader_clean(
    dataset_path: str,
    split: str,
    config: BertConfig,
) -> DataLoader:
    """Clean pipeline using streaming approach for large datasets."""
    return get_split_dataloader_streaming(dataset_path, split, config)


def get_dataset(
    dataset_path: str,
    dataset_name: str | None,
    split: str,
) -> Dataset:
    dataset = load_dataset(path=dataset_path, name=dataset_name, split=split)
    return dataset  # type: ignore[return-value]


class StreamingBertDataset(torch.utils.data.IterableDataset):
    """
    STREAMING SOLUTION: Process data on-the-fly while maintaining exact batch size.

    Key idea: Buffer training instances until we have exactly `batch_size` samples,
    then yield them. This gives us streaming processing + exact batch control.
    """

    def __init__(
        self,
        dataset_path: str,
        split: str,
        config: BertConfig,
        buffer_size: int = 1000,
    ) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.split = split
        self.config = config
        self.buffer_size = buffer_size

        # Load the raw dataset as streaming
        # For streaming datasets, we need to handle split parsing
        if ":" in split and "[" in split:
            # Parse "train[:50]" format
            base_split = split.split("[")[0]  # "train"
            slice_part = split.split("[")[1].split("]")[0]  # ":50"
            self.raw_dataset = load_dataset(dataset_path, split=base_split, streaming=True)
            if ":" in slice_part and slice_part.split(":")[1]:
                # Take only the specified number
                num_samples = int(slice_part.split(":")[1])
                self.raw_dataset = self.raw_dataset.take(num_samples)
        else:
            self.raw_dataset = load_dataset(dataset_path, split=split, streaming=True)

        # We'll create documents_dataset lazily when needed
        self._documents_dataset = None
        self.vocab_words = list(bert_tokenizer.get_vocab().keys())
        self.rng = random.Random(42)

    def __iter__(self):
        """Stream training instances while maintaining exact batch sizes."""
        # CRITICAL: For multi-worker support, each worker should process different data
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Multiple workers: skip data to avoid duplication
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            # Skip entries for this worker
            dataset_iter = iter(self.raw_dataset)
            for _ in range(worker_id):
                next(dataset_iter, None)
            # Take every num_workers-th item
            filtered_dataset = []
            for i, item in enumerate(dataset_iter):
                if i % num_workers == 0:
                    filtered_dataset.append(item)
                    if len(filtered_dataset) >= 100:  # Process in chunks
                        break
            raw_dataset = filtered_dataset
        else:
            # Single worker: use all data
            raw_dataset = self.raw_dataset

        instance_buffer = []
        documents_dataset = self._get_documents_dataset()

        for raw_item in raw_dataset:
            # Convert text to documents
            documents = text_to_documents(raw_item["text"])

            # Create training instances from each document
            for document in documents:
                instances = create_pretraining_examples_from_documents(
                    documents_dataset,
                    document,
                    max_seq_length=self.config.max_seq_length,
                    short_seq_prob=self.config.short_seq_prob,
                    masked_lm_prob=self.config.masked_lm_prob,
                    max_predictions_per_seq=self.config.max_predictions_per_seq,
                    vocab_words=self.vocab_words,
                    rng=self.rng,
                )

                # Add to buffer
                instance_buffer.extend(instances)

                # Yield batches when buffer is large enough
                while len(instance_buffer) >= self.config.batch_size:
                    batch_instances = instance_buffer[: self.config.batch_size]
                    instance_buffer = instance_buffer[self.config.batch_size :]

                    # Convert to the format expected by simple_collate_fn
                    yield from batch_instances
        super().__init__()
        self.dataset_path = dataset_path
        self.split = split
        self.config = config
        self.buffer_size = buffer_size

        # Load the raw dataset as streaming
        # For streaming, we need to handle split parsing
        if ":" in split and "[" in split:
            # Parse "train[:50]" format
            base_split = split.split("[")[0]  # "train"
            slice_part = split.split("[")[1].split("]")[0]  # ":50"
            self.raw_dataset = load_dataset(dataset_path, split=base_split, streaming=True)
            if ":" in slice_part and slice_part.split(":")[1]:
                # Take only the specified number
                num_samples = int(slice_part.split(":")[1])
                self.raw_dataset = self.raw_dataset.take(num_samples)
        else:
            self.raw_dataset = load_dataset(dataset_path, split=split, streaming=True)

        # We'll create documents_dataset lazily when needed
        self._documents_dataset = None
        self.vocab_words = list(bert_tokenizer.get_vocab().keys())
        self.rng = random.Random(42)

    def _get_documents_dataset(self):
        """Lazy loading of documents dataset for NSP random sampling."""
        if self._documents_dataset is None:
            # For streaming, we need a small sample for NSP random document selection
            # Take first 1000 examples and preprocess them
            sample_dataset = self.raw_dataset.take(1000)
            sample_list = list(sample_dataset)

            # Create documents from sample
            all_docs = []
            for item in sample_list:
                docs = text_to_documents(item["text"])
                all_docs.extend(docs)

            # Create a simple dataset-like object for NSP
            self._documents_dataset = {"document": all_docs}

        return self._documents_dataset

    def __iter__(self):
        """Stream training instances while maintaining exact batch sizes."""
        instance_buffer = []
        documents_dataset = self._get_documents_dataset()

        for raw_item in self.raw_dataset:
            # Convert text to documents
            documents = text_to_documents(raw_item["text"])

            # Create training instances from each document
            for document in documents:
                instances = create_pretraining_examples_from_documents(
                    documents_dataset,
                    document,
                    max_seq_length=self.config.max_seq_length,
                    short_seq_prob=self.config.short_seq_prob,
                    masked_lm_prob=self.config.masked_lm_prob,
                    max_predictions_per_seq=self.config.max_predictions_per_seq,
                    vocab_words=self.vocab_words,
                    rng=self.rng,
                )

                # Add to buffer
                instance_buffer.extend(instances)

                # Yield batches when buffer is large enough
                while len(instance_buffer) >= self.config.batch_size:
                    batch_instances = instance_buffer[: self.config.batch_size]
                    instance_buffer = instance_buffer[self.config.batch_size :]

                    # Convert to the format expected by simple_collate_fn
                    for instance in batch_instances:
                        yield instance


def get_split_dataloader_streaming(
    dataset_path: str,
    split: str,
    config: BertConfig,
) -> DataLoader:
    """
    STREAMING SOLUTION: Process data on-the-fly with exact batch size control.

    Benefits:
    - No upfront preprocessing (starts immediately)
    - Exact batch size control
    - Memory efficient
    - Works with datasets of any size
    - GPU OPTIMIZATION: Prefetching + background workers for full GPU utilization
    """

    streaming_dataset = StreamingBertDataset(
        dataset_path=dataset_path,
        split=split,
        config=config,
    )

    # IMPORTANT: For IterableDataset, we need to be careful with num_workers
    # If the dataset has internal randomness, use num_workers=0
    # For production with deterministic processing, can use multiple workers

    dataloader = DataLoader(
        streaming_dataset,
        batch_size=config.batch_size,
        collate_fn=simple_collate_fn,
        num_workers=config.num_workers,  # Background data loading for GPU efficiency
        prefetch_factor=4,  # Each worker prefetches 4 batches ahead
        pin_memory=True,  # Fast GPU transfer
        persistent_workers=True,  # Keep workers alive between epochs
        drop_last=True,  # Consistent batch sizes
    )

    return dataloader


def simple_collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """
    BEST PRACTICE: Simple collate function that only handles tokenization and padding.
    No complex transformations - just convert tokens to IDs and pad.
    """
    batch_tokens = []
    batch_segment_ids = []
    batch_is_random_next = []
    batch_masked_lm_labels = []

    pad_id = bert_tokenizer.token_to_id("[PAD]")
    max_len = max(len(item["tokens"]) for item in batch)

    for item in batch:
        # Convert tokens to IDs
        token_ids = [bert_tokenizer.token_to_id(token) for token in item["tokens"]]
        batch_tokens.append(torch.tensor(token_ids))

        # Segment IDs
        batch_segment_ids.append(torch.tensor(item["segment_ids"]))

        # NSP labels
        batch_is_random_next.append(item["is_random_next"])

        # MLM labels - create padded tensor
        mlm_labels = torch.full((max_len,), pad_id, dtype=torch.long)
        for i, pos in enumerate(item["masked_lm_positions"]):
            mlm_labels[pos] = bert_tokenizer.token_to_id(item["masked_lm_labels"][i])
        batch_masked_lm_labels.append(mlm_labels)

    # Pad sequences
    tokens_padded = pad_sequence(batch_tokens, padding_value=pad_id, batch_first=True)
    segment_ids_padded = pad_sequence(batch_segment_ids, padding_value=0, batch_first=True)

    return {
        "tokens": tokens_padded,
        "segment_ids": segment_ids_padded,
        "is_random_next": torch.tensor(batch_is_random_next, dtype=torch.long),
        "masked_lm_labels": torch.stack(batch_masked_lm_labels),
    }


def get_split_dataloader_clean(
    dataset_path: str,
    split: str,
    config: BertConfig,
) -> DataLoader:
    """Clean pipeline using streaming approach for large datasets."""
    return get_split_dataloader_streaming(dataset_path, split, config)


if __name__ == "__main__":
    config = BertConfig()

    print("=== Testing STREAMING Pipeline (Best for Large Datasets) ===")
    print("On-the-fly processing → No upfront preprocessing → Immediate start")

    # Streaming solution - starts immediately, no preprocessing delay
    print("Creating streaming dataloader...")
    streaming_dataloader = get_split_dataloader_streaming(
        config.dataset_path,
        "train[:50]",  # Larger sample to show streaming benefits
        config,
    )
    print("✓ Streaming DataLoader created successfully (immediate start!)")

    # Test batch size consistency and immediate data flow
    batch_sizes = []
    print("Testing streaming batch consistency...")
    start_time = time.time()

    for i, batch in enumerate(streaming_dataloader):
        batch_size = batch["tokens"].shape[0]
        batch_sizes.append(batch_size)
        print(f"Streaming Batch {i + 1}: size={batch_size}")
        if i >= 4:  # Check first 5 batches to show consistency
            break

    streaming_time = time.time() - start_time
    print(f"Streaming approach time for 5 batches: {streaming_time:.2f}s")

    print("\n" + "=" * 60)
    print("STREAMING SOLUTION FOR LARGE DATASETS")
    print("=" * 60)

    print("\n� Key Benefits of Streaming Approach:")
    print("  • IMMEDIATE START - No preprocessing delay")
    print("  • EXACT BATCH SIZE - Every batch is exactly", config.batch_size)
    print("  • MEMORY EFFICIENT - Only loads what's needed")
    print("  • SCALES TO ANY SIZE - Works with TB+ datasets")
    print("  • REAL STREAMING - Processes on-the-fly")

    print(f"\n📊 Batch Size Verification:")
    print(f"  • Target batch size: {config.batch_size}")
    print(f"  • Actual batch sizes: {batch_sizes}")
    print(f"  • All batches correct: {all(size == config.batch_size for size in batch_sizes)}")

    print(f"\n⚡ Performance for Large Datasets:")
    print(f"  • Streaming: Starts immediately, continuous processing")
    print(f"  • Preprocessing: Would take hours/days for full BookCorpus")
    print(f"  • Winner: Streaming (essential for production!)")
