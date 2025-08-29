import marimo

__generated_with = "0.15.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import random
    import collections
    from dataclasses import dataclass

    from datasets import Dataset, DatasetDict, load_dataset

    from toynlp.bert.tokenizer import BertTokenizer
    return BertTokenizer, Dataset, collections, load_dataset, random


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load Tokenizer""")
    return


@app.cell
def _(BertTokenizer):
    bert_tokenizer = BertTokenizer().load()

    # Test target language tokenizer
    text = "Two men are at the stove preparing food and vibecoding."
    output = bert_tokenizer.encode(text)
    print(f"Text: {text}")
    print(f"Tokens: {output.tokens}")
    print(f"Type Ids: {output.type_ids}")
    return (bert_tokenizer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load Raw Dataset""")
    return


@app.cell
def _(load_dataset):
    dataset_path: str = "lucadiliello/bookcorpusopen"
    dataset_name: str | None = None

    raw_dataset = load_dataset(dataset_path, dataset_name, split="train[:10]")
    raw_dataset
    return (raw_dataset,)


@app.cell
def _(raw_dataset):
    print(raw_dataset["text"][0][:10000])
    return


@app.cell
def _(raw_dataset):
    raw_dataset["text"][0].split("\n")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Data processing""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Get all documents

    Ref: https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/create_pretraining_data.py#L179
    """
    )
    return


@app.cell
def _(bert_tokenizer, random):
    # https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/tokenization.py#L78
    def convert_to_unicode(text):
        """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))


    def text_to_documents(text: str) -> list[list[str]]:
        all_documents = [[]]
        lines = text.split("\n")
        for line in lines:
            line = convert_to_unicode(line).strip()
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
                all_documents.append(doc)
        return all_documents
    return batch_text_to_documents, text_to_documents


@app.cell
def _(raw_dataset, text_to_documents):
    text_to_documents(raw_dataset["text"][0])
    return


@app.cell
def _(batch_text_to_documents, raw_dataset):
    len(batch_text_to_documents(raw_dataset["text"]))
    return


@app.cell
def _(batch_text_to_documents, raw_dataset):
    # https://huggingface.co/docs/datasets/en/about_map_batch#input-size--output-size
    documents_dataset = raw_dataset.map(
        lambda batch: {"document": batch_text_to_documents(batch["text"])},
        batched=True,
        # batch_size=4,
        remove_columns=["text", "title"],
    )
    len(documents_dataset)
    return (documents_dataset,)


@app.cell
def _(documents_dataset):
    # for i in range(10000):
    #     if len(documents_dataset["document"][i]) > 1:
    #         print(i, len(documents_dataset["document"][i]))

    documents_dataset["document"][8878]
    return


@app.cell
def _(documents_dataset):
    type(documents_dataset)
    return


@app.function
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


@app.cell
def _(collections):
    MaskedLmInstance = collections.namedtuple(
        "MaskedLmInstance", ["index", "label"]
    )


    def create_masked_lm_predictions(
        tokens,
        masked_lm_prob,
        max_predictions_per_seq,
        vocab_words,
        rng,
        do_whole_word_mask=False,
    ):
        """Creates the predictions for the masked LM objective."""

        cand_indexes = []
        for i, token in enumerate(tokens):
            if token == "[CLS]" or token == "[SEP]":
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
            if (
                do_whole_word_mask
                and len(cand_indexes) >= 1
                and token.startswith("##")
            ):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        rng.shuffle(cand_indexes)

        output_tokens = list(tokens)

        num_to_predict = min(
            max_predictions_per_seq,
            max(1, int(round(len(tokens) * masked_lm_prob))),
        )

        masked_lms = []
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
                    masked_token = "[MASK]"
                else:
                    # 10% of the time, keep original
                    if rng.random() < 0.5:
                        masked_token = tokens[index]
                    # 10% of the time, replace with random word
                    else:
                        masked_token = vocab_words[
                            rng.randint(0, len(vocab_words) - 1)
                        ]

                output_tokens[index] = masked_token

                masked_lms.append(
                    MaskedLmInstance(index=index, label=tokens[index])
                )
        assert len(masked_lms) <= num_to_predict
        masked_lms = sorted(masked_lms, key=lambda x: x.index)

        masked_lm_positions = []
        masked_lm_labels = []
        for p in masked_lms:
            masked_lm_positions.append(p.index)
            masked_lm_labels.append(p.label)

        return (output_tokens, masked_lm_positions, masked_lm_labels)
    return (create_masked_lm_predictions,)


@app.cell
def _(Dataset, create_masked_lm_predictions, random):
    def create_pretraining_examples_from_documents(
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
                            random_document_index = rng.randint(
                                0, len(documents_dataset) - 1
                            )
                            # if random_document_index != document_index:
                            if (
                                documents_dataset["document"][
                                    random_document_index
                                ][0]
                                != document
                            ):
                                break

                        random_document = documents_dataset["document"][
                            random_document_index
                        ]
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

                    (tokens, masked_lm_positions, masked_lm_labels) = (
                        create_masked_lm_predictions(
                            tokens,
                            masked_lm_prob,
                            max_predictions_per_seq,
                            vocab_words,
                            rng,
                        )
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
    return (create_pretraining_examples_from_documents,)


@app.cell
def _(
    bert_tokenizer,
    create_pretraining_examples_from_documents,
    documents_dataset,
    random,
):
    instances = create_pretraining_examples_from_documents(
        documents_dataset,
        documents_dataset["document"][0],
        max_seq_length=128,
        short_seq_prob=0.1,
        masked_lm_prob=0.15,
        max_predictions_per_seq=20,
        vocab_words=list(bert_tokenizer.get_vocab().keys()),
        rng=random.Random(12345),
    )

    len(instances)
    return (instances,)


@app.cell
def _(instances):
    instance = instances[0]
    print(instance)

    # print(f"Tokens: {instance.tokens}\n")
    # print(f"Segment Ids: {instance.segment_ids}\n")
    # print(f"Is Random Next: {instance.is_random_next}\n")
    # print(f"Masked LM Positions: {instance.masked_lm_positions}\n")
    return


@app.cell
def _(
    Dataset,
    bert_tokenizer,
    create_pretraining_examples_from_documents,
    documents_dataset,
    random,
):
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


    simple_batch_instances = batch_create_pretraining_examples_from_documents(
        documents_dataset,
        documents_dataset["document"][:3],
        max_seq_length=128,
        short_seq_prob=0.1,
        masked_lm_prob=0.15,
        max_predictions_per_seq=20,
        vocab_words=list(bert_tokenizer.get_vocab().keys()),
        rng=random.Random(12345),
    )

    simple_batch_instances
    return (batch_create_pretraining_examples_from_documents,)


@app.cell
def _(documents_dataset):
    len(documents_dataset)
    return


@app.cell
def _(
    batch_create_pretraining_examples_from_documents,
    bert_tokenizer,
    documents_dataset,
    random,
):
    all_pretrain_instances = documents_dataset.map(
        lambda batch: {
            "instances": batch_create_pretraining_examples_from_documents(
                documents_dataset,
                batch["document"],  # TODO: batch or batch["document"] ???
                max_seq_length=128,
                short_seq_prob=0.1,
                masked_lm_prob=0.15,
                max_predictions_per_seq=20,
                vocab_words=list(bert_tokenizer.get_vocab().keys()),
                rng=random.Random(12345),
            )
        },
        batched=True,
        batch_size=100,
        remove_columns=["document"],
    )

    all_pretrain_instances
    return (all_pretrain_instances,)


@app.cell
def _(all_pretrain_instances):
    all_pretrain_instances["instances"][0]
    return


@app.cell
def _(all_pretrain_instances):
    is_next_count, not_next_count = 0, 0
    for instance_item in all_pretrain_instances:
        if instance_item["instances"]["is_random_next"]:
            not_next_count += 1
        else:
            is_next_count += 1

    print(f"Is Next: {is_next_count}, Not Next: {not_next_count}")
    return


@app.cell
def _(all_pretrain_instances, documents_dataset):
    len(all_pretrain_instances) - len(documents_dataset)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
