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

    from datasets import DatasetDict, load_dataset

    from toynlp.bert.tokenizer import BertTokenizer
    return BertTokenizer, load_dataset, random


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
    all_documents = raw_dataset.map(
        lambda batch: {"all_documents": batch_text_to_documents(batch["text"])},
        batched=True,
        # batch_size=4,
        remove_columns=["text"],
    )
    len(all_documents)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
