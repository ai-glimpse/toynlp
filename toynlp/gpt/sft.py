from typing import Any
from collections.abc import Iterable
from datasets import load_dataset, Dataset, concatenate_datasets
from torch import nn
import torch
import math
from toynlp.gpt.model import GPTModel
from toynlp.gpt.tokenizer import GPTTokenizer
from tokenizers import Tokenizer
from toynlp.gpt.train import GPTTrainer
from toynlp.gpt.config import GPTConfig
import wandb
from toynlp.paths import GPT_MODEL_PATH, GPT_SFT_MODEL_PATH


class SftDataset:
    def __init__(
        self,
        dataset_names: str | Iterable[str] | None = None,
        split: str = "train",
        eos_token: str = "___",  # noqa: S107
    ) -> None:
        if dataset_names is None:
            dataset_names = ["databricks/databricks-dolly-15k", "teknium/GPT4-LLM-Cleaned"]
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        self.dataset_names = list(dataset_names)
        self.split = split
        self.eos_token = eos_token
        self.raw_dataset = self._load_raw_dataset(self.dataset_names, split)

    def load_sft_dataset(self) -> Dataset:
        return self.raw_dataset.map(self._dataset_transform, remove_columns=self.raw_dataset.column_names, num_proc=4)

    def _load_raw_dataset(self, dataset_names: list[str], split: str) -> Dataset:
        datasets = [self._load_dataset_by_name(name, split) for name in dataset_names]
        if len(datasets) == 1:
            return datasets[0]
        return concatenate_datasets(datasets)

    def _load_dataset_by_name(self, dataset_name: str, split: str) -> Dataset:
        if dataset_name == "databricks/databricks-dolly-15k":
            return self._load_dolly_dataset(split)
        if dataset_name == "teknium/GPT4-LLM-Cleaned":
            return self._load_gpt4_llm_dataset(split)
        # fallback to raw load for any additional datasets
        return load_dataset(dataset_name, split=split)

    def _load_dolly_dataset(self, split: str) -> Dataset:
        dataset = load_dataset("databricks/databricks-dolly-15k", split=split)
        return dataset

    def _load_gpt4_llm_dataset(self, split: str) -> Dataset:
        dataset = load_dataset("teknium/GPT4-LLM-Cleaned", split=split)

        def _transform(row: dict[str, Any]) -> dict[str, Any]:
            return {
                "instruction": row.get("instruction") or "",
                "context": row.get("input") or "",
                "response": row.get("output") or "",
            }

        standard_cols = [col for col in dataset.column_names if col not in {"instruction", "context", "response"}]
        return dataset.map(_transform, remove_columns=standard_cols)

    def _dataset_transform(self, row: dict[str, Any]) -> dict[str, Any]:
        prompt_text = self._extract_prompt(row)
        context = self._extract_context(row)
        response = self._extract_response(row)
        prompt = prompt_text or ""
        if context:
            input_text = self.template(with_context=True).format(prompt=prompt, context=context, response=response)
        else:
            input_text = self.template(with_context=False).format(prompt=prompt, response=response)
        return {"input_text": input_text}

    def _extract_prompt(self, row: dict[str, Any]) -> str:
        return row.get("instruction") or row.get("prompt") or row.get("question") or ""

    def _extract_context(self, row: dict[str, Any]) -> str:
        return row.get("context") or row.get("system_prompt") or ""

    def _extract_response(self, row: dict[str, Any]) -> str:
        return row.get("response") or row.get("output") or row.get("completion") or row.get("answer") or ""

    def template(self, with_context: bool = False) -> str:
        if with_context:
            return """Human: {prompt}\n\nContext: {context}\n\nAssistant: {response}""" + self.eos_token
        return """Human: {prompt}\n\nAssistant: {response}""" + self.eos_token


def text_to_token_ids(
    texts: list[str],
    tokenizer: Tokenizer,
    max_length: int,
) -> dict[str, list[torch.Tensor]]:
    input_ids = []
    labels = []
    pad_id = tokenizer.token_to_id("<pad>")
    for text in texts:
        token_ids = tokenizer.encode(text).ids[:max_length]
        # use re find "Assistant:" position
        if len(token_ids) < max_length:
            token_ids += [pad_id] * (max_length - len(token_ids))
        input_id = torch.tensor(token_ids, dtype=torch.long)
        input_ids.append(input_id)
        label = input_id.clone()
        assistant_marker = "Assistant:"
        assistant_pos = text.find(assistant_marker)
        if assistant_pos == -1:
            prompt_token_count = 0
        else:
            prompt_text = text[: assistant_pos + len(assistant_marker)]
            prompt_token_count = len(tokenizer.encode(prompt_text).ids[:max_length])
        label[:prompt_token_count] = pad_id  # ignore prompt tokens in loss calculation
        labels.append(label)
    return {"input_ids": input_ids, "labels": labels}


def get_sft_dataloaders(
    config: GPTConfig,
    gpt_tokenizer: Tokenizer,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    sft_dataset = SftDataset().load_sft_dataset()
    sft_token_dataset = sft_dataset.map(
        lambda batch: text_to_token_ids(batch["input_text"], gpt_tokenizer, config.max_seq_length),
        remove_columns=["input_text"],
        batched=True,
        num_proc=4,
    )

    # Split into train, val, test
    total_size = len(sft_token_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    print(f"Total dataset size: {total_size}, train size: {train_size}, val size: {val_size}")

    # train_size = 24
    # val_size = 24
    # total_size = train_size + val_size + 24

    train_dataset = sft_token_dataset.select(range(train_size))
    val_dataset = sft_token_dataset.select(range(train_size, train_size + val_size))
    test_dataset = sft_token_dataset.select(range(train_size + val_size, total_size))

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset.with_format("torch"),
        # dataset=train_dataset,
        batch_size=config.batch_size,
        num_workers=4,
        prefetch_factor=4,
        drop_last=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset.with_format("torch"),
        batch_size=config.batch_size,
        num_workers=4,
        prefetch_factor=4,
        drop_last=True,
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset.with_format("torch"),
        batch_size=config.batch_size,
        num_workers=4,
        prefetch_factor=4,
        drop_last=True,
    )

    return train_dataloader, val_dataloader, test_dataloader


def mark_only_lora_as_trainable(model: GPTModel) -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
        else:
            p.requires_grad = True


def apply_lora(
    model: GPTModel, r: int = 16, alpha: int = 32, dropout: float = 0.05, target_modules: list[str] | None = None
) -> GPTModel:
    if target_modules is None:
        target_modules = ["causal_mha", "ffn", "lm_head"]
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            if hasattr(module, "lora_adapter"):
                continue  # already patched
            lora = LoRALayer(module.in_features, module.out_features, r, alpha, dropout)
            lora.to(device=model.device)
            module.add_module("lora_adapter", lora)
            if not hasattr(module, "_original_forward"):
                module._original_forward = module.forward  # noqa: SLF001

            def lora_forward(x, orig_forward=module._original_forward, lora_layer=lora) -> torch.Tensor:  # noqa: SLF001
                base_out = orig_forward(x)
                return base_out + lora_layer(x)

            module.forward = lora_forward
    return model


class LoRALayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int, alpha: int, dropout: float) -> None:
        super().__init__()
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.dropout(x @ self.lora_A.T) @ self.lora_B.T * self.scaling


class GPTSFTTrainer(GPTTrainer):
    def save_model(self):
        """Save the current model to the specified path."""
        torch.save(self._merge_lora_state_dict(), self.model_path)

    def _merge_lora_state_dict(self) -> dict[str, torch.Tensor]:
        merged_state = {
            name: param.clone() for name, param in self.model.state_dict().items() if "lora_adapter" not in name
        }

        for module_name, module in self.model.named_modules():
            lora_adapter = getattr(module, "lora_adapter", None)
            if not module_name or not isinstance(lora_adapter, LoRALayer):
                continue

            delta_weight = (lora_adapter.lora_B @ lora_adapter.lora_A) * lora_adapter.scaling
            weight_key = f"{module_name}.weight"
            if weight_key not in merged_state:
                continue
            merged_state[weight_key] = merged_state[weight_key] + delta_weight.to(merged_state[weight_key].device)

        return merged_state


def train_model(config: GPTConfig) -> None:
    """Train the GPT model with the given configuration."""
    if config.wandb_enabled:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_name,
            config=config.to_dict(),
        )

    tokenizer = GPTTokenizer().load()

    train_dataloader, val_dataloader, test_dataloader = get_sft_dataloaders(
        config=config,
        gpt_tokenizer=tokenizer,
    )

    padding_token_id = tokenizer.token_to_id("<pad>")
    model = GPTModel(config, padding_idx=padding_token_id)
    # TODO: load pre-trained model weights before SFT
    model.load_state_dict(torch.load(GPT_MODEL_PATH, map_location=model.device))
    # apply lora
    model = apply_lora(model)
    mark_only_lora_as_trainable(model)

    trainer = GPTSFTTrainer(config=config, pad_token_id=padding_token_id, model=model, model_path=GPT_SFT_MODEL_PATH)
    trainer.base_lr = 1e-4  # set a different base learning rate for SFT
    trainer.config.epochs = 100  # set a smaller number of epochs for SFT
    trainer.train(train_dataloader, val_dataloader, test_dataloader)


if __name__ == "__main__":
    from toynlp.gpt.config import GPTConfig

    config = GPTConfig(
        wandb_enabled=True,
        wandb_name="gpt_sft_lora_r16_alpha32",
    )
    tokenizer = GPTTokenizer().load()

    # dataset = Dataset()
    # sft_dataset = dataset.load_sft_dataset()
    # for i in range(3):
    #     print(sft_dataset[i]["input_text"])
    #     print("---"*20)

    # train_dataloader, val_dataloader, test_dataloader = get_sft_dataloaders(
    #     config=config,
    #     gpt_tokenizer=tokenizer,
    # )
    # for batch in train_dataloader:
    #     for item in batch["input_ids"]:
    #         print("Item shape:", item.shape)
    #         print(tokenizer.decode(item.tolist(), skip_special_tokens=False))
    #         print("-" * 20)
    #         break

    train_model(config)
