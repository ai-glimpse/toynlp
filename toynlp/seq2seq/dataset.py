import torch
from datasets import DatasetDict, load_dataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from toynlp.seq2seq.config import DataConfig


def get_dataset(
    dataset_path: str,
    dataset_name: str | None,
) -> DatasetDict:
    dataset = load_dataset(path=dataset_path, name=dataset_name)
    return dataset  # type: ignore[return-value]


def collate_fn(
    batch: dict[str, list[str]],
    source_tokenizer: Tokenizer,
    target_tokenizer: Tokenizer,
    max_length: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_input = []
    batch_target = []
    input_pad_id = source_tokenizer.token_to_id("[PAD]")
    target_pad_id = target_tokenizer.token_to_id("[PAD]")

    for item in batch:
        src_tensor = source_tokenizer.encode(item[source_lang]).ids  # type: ignore[call-arg,index]
        tgt_tensor = target_tokenizer.encode(item[target_lang]).ids  # type: ignore[call-arg,index]
        batch_input.append(torch.tensor(src_tensor[:max_length], dtype=torch.long))
        batch_target.append(torch.tensor(tgt_tensor[:max_length], dtype=torch.long))
    batch_input_tensor = pad_sequence(batch_input, padding_value=input_pad_id, batch_first=True)
    batch_target_tensor = pad_sequence(batch_target, padding_value=target_pad_id, batch_first=True)
    return batch_input_tensor, batch_target_tensor


def get_split_dataloader(
    dataset: DatasetDict,
    split: str,
    source_tokenizer: Tokenizer,
    target_tokenizer: Tokenizer,
    data_config: DataConfig,
) -> DataLoader:
    dataloader = DataLoader(
        dataset=dataset[split],  # type: ignore[arg-type]
        batch_size=data_config.batch_size,
        num_workers=data_config.num_workers,
        shuffle=data_config.shuffle,
        collate_fn=lambda batch: collate_fn(batch, source_tokenizer, target_tokenizer, data_config.max_length),  # type: ignore[arg-type]
        drop_last=True,
    )
    return dataloader


if __name__ == "__main__":
    from toynlp.seq2seq.config import get_config
    from toynlp.seq2seq.tokenizer import Seq2SeqTokenizer

    config = get_config()

    dataset = get_dataset(
        dataset_path=config.dataset.path,
        dataset_name=config.dataset.name,
    )

    source_tokenizer = Seq2SeqTokenizer(lang=config.dataset.source_lang).load()
    target_tokenizer = Seq2SeqTokenizer(lang=config.dataset.target_lang).load()
    train_dataloader = get_split_dataloader(dataset, "train", source_tokenizer, target_tokenizer, config.data)
    for batch_input, batch_target in train_dataloader:
        print(batch_input.shape, batch_target.shape)
        print(batch_input[0], batch_target[0])
        break
