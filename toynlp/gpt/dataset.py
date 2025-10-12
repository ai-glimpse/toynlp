from datasets import Dataset, load_dataset
from tokenizers import Tokenizer
import torch
from torch.utils.data import DataLoader

from toynlp.gpt.config import GPTConfig
from toynlp.gpt.tokenizer import GPTTokenizer


def get_dataset(
    dataset_path: str,
    dataset_name: str | None,
    split: str,
) -> Dataset:
    dataset = load_dataset(path=dataset_path, name=dataset_name, split=split, streaming=True)
    return dataset  # type: ignore[return-value]


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


def get_split_dataloader(
    dataset_path: str,
    split: str,
    config: GPTConfig,
) -> DataLoader:
    raw_dataset = load_dataset(path=dataset_path, name=None, split=split)
    raw_dataset = raw_dataset.shuffle(seed=42).to_iterable_dataset(num_shards=32)  # type: ignore[call-arg]
    context_dataset = raw_dataset.map(
        lambda batch: {
            "input_ids": split_text_into_contexts(
                batch["text"],
                config.max_seq_length,
                gpt_tokenizer,
            )
        },
        remove_columns=["title", "text"],
        batched=True,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=context_dataset,
        batch_size=config.batch_size,
        num_workers=8,
        prefetch_factor=4,
        drop_last=True,
    )

    return dataloader


if __name__ == "__main__":
    from tqdm import tqdm

    gpt_tokenizer = GPTTokenizer().load()

    config = GPTConfig()

    demo_dataset_loader = get_split_dataloader(
        config.dataset_path,
        "train[:1%]",
        config,
    )
    # print(f"Number of training batches: {len(val_dataset_loader)}")
    for i, batch in enumerate(tqdm(demo_dataset_loader)):
        # Process each batch
        # print(batch)
        print(batch["input_ids"].shape, batch["input_ids"].dtype)
        print(batch["input_ids"][0][:10])
        print("=" * 20, i, "=" * 20)
