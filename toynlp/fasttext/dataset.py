import torch
from datasets import DatasetDict, load_dataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from toynlp.fasttext.config import FastTextConfig


def get_dataset(
    dataset_path: str,
    dataset_name: str | None,
) -> DatasetDict:
    dataset = load_dataset(path=dataset_path, name=dataset_name)
    return dataset  # type: ignore[return-value]


def collate_fn(
    batch: dict[str, list[str]],
    tokenizer: Tokenizer,
    max_length: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_input = []
    batch_target = []
    pad_id = tokenizer.token_to_id("[PAD]")

    for item in batch:
        text_tensor = tokenizer.encode(item["text"]).ids  # type: ignore[call-arg,index]
        batch_input.append(torch.tensor(text_tensor[:max_length], dtype=torch.long))
        batch_target.append(torch.tensor(item["label"], dtype=torch.long))  # type: ignore[arg-type]
    batch_input_tensor = pad_sequence(batch_input, padding_value=pad_id, batch_first=True)
    batch_target_tensor = torch.tensor(batch_target, dtype=torch.long)
    return batch_input_tensor, batch_target_tensor


def get_split_dataloader(
    dataset: DatasetDict,
    split: str,
    tokenizer: Tokenizer,
    config: FastTextConfig,
) -> DataLoader:
    dataloader = DataLoader(
        dataset=dataset[split],  # type: ignore[arg-type]
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=config.shuffle,
        collate_fn=lambda batch: collate_fn(
            batch,  # type: ignore[arg-type]
            tokenizer,
            config.max_length,
        ),
    )
    return dataloader


if __name__ == "__main__":
    from toynlp.fasttext.config import create_config_from_cli
    from toynlp.fasttext.tokenizer import FastTextTokenizer

    config = create_config_from_cli()

    dataset = get_dataset(
        dataset_path=config.dataset_path,
        dataset_name=config.dataset_name,
    )

    tokenizer = FastTextTokenizer().load()
    train_dataloader = get_split_dataloader(dataset, "train", tokenizer, config)
    for batch_input, batch_target in train_dataloader:
        print(batch_input.shape, batch_target.shape)
        # print(batch_input[0], batch_target[0])
        # break
