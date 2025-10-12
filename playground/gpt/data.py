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

    from toynlp.gpt.tokenizer import GPTTokenizer
    from tokenizers import Tokenizer
    import torch
    return GPTTokenizer, Tokenizer, load_dataset, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load Tokenizer""")
    return


@app.cell
def _(GPTTokenizer):
    gpt_tokenizer = GPTTokenizer().load()

    # Test target language tokenizer
    text = "Two men are at the stove preparing food and vibecoding."
    output = gpt_tokenizer.encode(text)
    print(f"Text: {text}")
    print(f"Tokens: {output.tokens}")
    print(f"Type Ids: {output.type_ids}")
    return (gpt_tokenizer,)


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
    print(raw_dataset["text"][0])
    print(len(raw_dataset["text"][0]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Data processing""")
    return


@app.cell
def _(Tokenizer, torch):
    def split_text_into_contexts(texts: str, max_length: int, tokenizer: Tokenizer) -> list[torch.Tensor]:
        contexts = []
        for text in texts:
            token_ids = tokenizer.encode(text).ids  # type: ignore[call-arg,index]
            for i in range(len(token_ids) // max_length + 1):
                start_idx = i * max_length
                end_idx = (i + 1) * max_length
                if end_idx < len(token_ids):
                    contexts.append(torch.tensor(token_ids[start_idx:end_idx], dtype=torch.long))
        return contexts
    return (split_text_into_contexts,)


@app.cell
def _(gpt_tokenizer, raw_dataset, split_text_into_contexts):
    context_dataset = raw_dataset.map(
        lambda batch: {
            "input_ids": split_text_into_contexts(
                batch["text"],
                512,
                gpt_tokenizer,
            )
        },
        remove_columns=["title", "text"],
        batched=True,
        num_proc=8,
    )

    context_dataset
    return (context_dataset,)


@app.cell
def _(context_dataset):
    context_dataset["input_ids"][0]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
