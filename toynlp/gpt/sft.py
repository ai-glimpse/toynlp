from typing import Any, cast
from collections.abc import Iterable
import pathlib
import random

from datasets import load_dataset, Dataset, concatenate_datasets
from torch import nn
from torch.utils.data import DataLoader
import torch
import math
from toynlp.gpt.model import GPTModel
from toynlp.gpt.tokenizer import GPTTokenizer
from tokenizers import Tokenizer
from toynlp.gpt.train import GPTTrainer
from toynlp.gpt.config import GPTConfig
import wandb
from toynlp.paths import GPT_SFT_MODEL_PATH, GPT_MODEL_PATH
from toynlp.util import current_device


class SftDataset:
    def __init__(
        self,
        dataset_names: str | Iterable[str] | None = None,
        split: str = "train",
        eos_token: str = "<eos>",  # noqa: S107
    ) -> None:
        if dataset_names is None:
            dataset_names = [
                "databricks/databricks-dolly-15k",
                "teknium/GPT4-LLM-Cleaned",
                "yahma/alpaca-cleaned",
            ]
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
        if dataset_name == "yahma/alpaca-cleaned":
            return self._load_alpaca_dataset(split)
        # fallback to raw load for any additional datasets
        return load_dataset(dataset_name, split=split)

    def _load_dolly_dataset(self, split: str) -> Dataset:
        dataset = cast("Dataset", load_dataset("databricks/databricks-dolly-15k", split=split))
        return dataset

    def _load_gpt4_llm_dataset(self, split: str) -> Dataset:
        dataset = cast("Dataset", load_dataset("teknium/GPT4-LLM-Cleaned", split=split))

        def _transform(row: dict[str, Any]) -> dict[str, Any]:
            return {
                "instruction": row.get("instruction") or "",
                "context": row.get("input") or "",
                "response": row.get("output") or "",
            }

        column_names = list(dataset.column_names)
        standard_cols = [col for col in column_names if col not in {"instruction", "context", "response"}]
        return dataset.map(_transform, remove_columns=standard_cols)

    def _load_alpaca_dataset(self, split: str) -> Dataset:
        dataset = cast("Dataset", load_dataset("yahma/alpaca-cleaned", split=split))

        def _transform(row: dict[str, Any]) -> dict[str, Any]:
            return {
                "instruction": row.get("instruction") or "",
                "context": row.get("input") or "",
                "response": row.get("output") or "",
            }

        column_names = list(dataset.column_names)
        standard_cols = [col for col in column_names if col not in {"instruction", "context", "response"}]
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
    dataset_names: list[str] | None = None,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Load SFT dataloaders for specified datasets."""
    sft_dataset = SftDataset(dataset_names=dataset_names).load_sft_dataset()
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

    train_dataset = sft_token_dataset.select(range(train_size))
    val_dataset = sft_token_dataset.select(range(train_size, train_size + val_size))
    test_dataset = sft_token_dataset.select(range(train_size + val_size, total_size))

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset.with_format("torch"),
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
    model: GPTModel, r: int = 16, alpha: int = 32, dropout: float = 0.1, target_modules: list[str] | None = None
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
        return self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling


class SFTTrainer(GPTTrainer):
    """SFT trainer with batch-level train loss logging and configurable metric prefix."""

    def __init__(
        self,
        config: GPTConfig,
        pad_token_id: int,
        model: GPTModel,
        model_path: pathlib.Path,
        metric_prefix: str = "sft",
    ) -> None:
        super().__init__(config, pad_token_id, model, model_path)
        self.metric_prefix = metric_prefix

    def train(
        self,
        train_dataloader: DataLoader[dict[str, torch.Tensor]],
        val_dataloader: DataLoader[dict[str, torch.Tensor]],
        test_dataloader: DataLoader[dict[str, torch.Tensor]],
    ) -> None:
        """Train with per-batch train loss logging."""
        best_val_loss = float("inf")
        for epoch in range(self.config.epochs):
            self.model.train()
            total_loss = 0.0
            total_samples = 0

            for batch in train_dataloader:
                self.update_lr()
                self.optimizer.zero_grad()

                batch_input_ids = batch["input_ids"].to(self.device)
                batch_target_ids = batch["labels"].to(self.device)

                input_ids = batch_input_ids[:, :-1]
                target_ids = batch_target_ids[:, 1:]

                logits = self.model(input_ids)
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

                # Log train batch loss
                if self.config.wandb_enabled:
                    wandb.log({f"{self.metric_prefix}/train_batch_loss": loss.item()})

                if random.random() < 0.01:
                    self._print_sample_predictions(input_ids[0], target_ids[0], logits[0], "train")

                loss.backward()
                if self.clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                self.optimizer.step()

                self.current_step += 1
                batch_size = input_ids.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

            avg_train_loss = total_loss / total_samples
            train_perplexity = torch.exp(torch.tensor(avg_train_loss)).item()

            # Validate (epoch-level only)
            self.model.eval()
            with torch.no_grad():
                val_loss_stats = self.calc_loss_loader(val_dataloader, "val")
                test_loss_stats = self.calc_loss_loader(test_dataloader, "test")

            current_lr = self.get_lr()
            print(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, Train PPL: {train_perplexity:.4f}, "
                f"Val Loss: {val_loss_stats['loss']:.4f}, Val PPL: {val_loss_stats['perplexity']:.4f}, "
                f"Test Loss: {test_loss_stats['loss']:.4f}, Test PPL: {test_loss_stats['perplexity']:.4f}, "
                f"LR: {current_lr:.6f}"
            )

            if val_loss_stats["loss"] < best_val_loss:
                best_val_loss = val_loss_stats["loss"]
                self.save_model()
                print(f"Saved best model at epoch {epoch + 1}")

            if self.config.wandb_enabled:
                wandb.log(
                    {
                        f"{self.metric_prefix}/epoch": epoch + 1,
                        f"{self.metric_prefix}/train_loss": avg_train_loss,
                        f"{self.metric_prefix}/train_perplexity": train_perplexity,
                        f"{self.metric_prefix}/val_loss": val_loss_stats["loss"],
                        f"{self.metric_prefix}/val_perplexity": val_loss_stats["perplexity"],
                        f"{self.metric_prefix}/test_loss": test_loss_stats["loss"],
                        f"{self.metric_prefix}/test_perplexity": test_loss_stats["perplexity"],
                        f"{self.metric_prefix}/learning_rate": current_lr,
                    }
                )


class GPTSFTTrainer(SFTTrainer):
    """SFT trainer with LoRA weight merging on save."""

    def __init__(
        self,
        config: GPTConfig,
        pad_token_id: int,
        model: GPTModel,
        model_path: pathlib.Path,
        metric_prefix: str = "sft",
    ) -> None:
        super().__init__(config, pad_token_id, model, model_path, metric_prefix=metric_prefix)

    def save_model(self) -> None:
        """Save the current model with merged LoRA weights."""
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
    """Train the GPT model with general SFT on diverse datasets."""
    if config.wandb_enabled:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_name,
            config=config.to_dict(),
        )

    print("\n" + "=" * 60)
    print("General SFT Training")
    print("=" * 60)

    tokenizer = GPTTokenizer().load()
    model = torch.load(GPT_MODEL_PATH, map_location=current_device, weights_only=False)

    # Load general datasets
    general_datasets = [
        "databricks/databricks-dolly-15k",
        "teknium/GPT4-LLM-Cleaned",
        "yahma/alpaca-cleaned",
    ]
    train_dataloader, val_dataloader, test_dataloader = get_sft_dataloaders(
        config=config,
        gpt_tokenizer=tokenizer,
        dataset_names=general_datasets,
    )

    padding_token_id = tokenizer.token_to_id("<pad>")

    # Apply LoRA
    model = apply_lora(model, r=16, alpha=32, dropout=0.1)
    mark_only_lora_as_trainable(model)

    trainer = GPTSFTTrainer(config=config, pad_token_id=padding_token_id, model=model, model_path=GPT_SFT_MODEL_PATH)
    trainer.base_lr = 1e-4
    trainer.config.epochs = 10
    trainer.train(train_dataloader, val_dataloader, test_dataloader)

    print("\n" + "=" * 60)
    print("General SFT training completed!")
    print("=" * 60)


if __name__ == "__main__":
    from toynlp.gpt.config import GPTConfig

    config = GPTConfig(
        wandb_enabled=True,
        wandb_name="gpt_sft_lora_general",
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
