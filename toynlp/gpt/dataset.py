from datasets import load_dataset
from tokenizers import Tokenizer
import torch
from torch.utils.data import DataLoader

from toynlp.gpt.config import GPTConfig
from toynlp.gpt.tokenizer import GPTTokenizer


def split_text_into_contexts(texts: list[str], max_length: int, tokenizer: Tokenizer) -> list[torch.Tensor]:
    contexts = []
    eos_id = tokenizer.token_to_id("<eos>")
    pad_id = tokenizer.token_to_id("<pad>")
    if eos_id is None or pad_id is None:
        msg = "Missing required special tokens <eos> or <pad> in tokenizer vocabulary"
        raise ValueError(msg)

    for text in texts:
        token_ids = tokenizer.encode(text).ids
        token_ids.append(eos_id)

        for start_idx in range(0, len(token_ids), max_length):
            chunk = token_ids[start_idx : start_idx + max_length]
            if len(chunk) < max_length:
                chunk.extend([pad_id] * (max_length - len(chunk)))
            contexts.append(torch.tensor(chunk, dtype=torch.long))

    return contexts


def get_split_dataloader(
    dataset_path: str,
    split: str,
    config: GPTConfig,
    gpt_tokenizer: Tokenizer,
) -> DataLoader:
    raw_dataset = load_dataset(path=dataset_path, split=split)
    num_shards = min(32, raw_dataset.num_rows)  # type: ignore[possible-missing-attribute]
    raw_dataset = raw_dataset.shuffle(seed=42).to_iterable_dataset(num_shards=num_shards)  # type: ignore[call-arg]
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
        prefetch_factor=8,
        drop_last=True,
    )

    return dataloader


if __name__ == "__main__":
    from tqdm import tqdm

    gpt_tokenizer = GPTTokenizer().load()

    config = GPTConfig()

    demo_dataset_loader = get_split_dataloader(
        config.dataset_path,
        "train[:1]",
        config,
        gpt_tokenizer,
    )
    # print(f"Number of training batches: {len(val_dataset_loader)}")
    for _, batch in enumerate(tqdm(demo_dataset_loader)):
        # Process each batch
        print(batch["input_ids"].shape, batch["input_ids"].dtype)

    # for j in range(len(batch["input_ids"])):
    #     print(gpt_tokenizer.decode(batch["input_ids"][j].tolist(), skip_special_tokens=False))
    #     print("-" * 20, j, "-" * 20)
