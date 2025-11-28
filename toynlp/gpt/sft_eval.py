"""Evaluation helpers for supervised fine-tuned GPT on ARC datasets."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

from git import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

from datasets import Dataset, load_dataset
from rich.console import Console
from rich.table import Table

from toynlp.gpt.config import GPTConfig
from toynlp.gpt.inference import GPTInference
from toynlp.paths import GPT_SFT_MODEL_PATH


@dataclass
class ARCEvalResult:
    accuracy: float
    total: int
    correct: int


class ARCEval:
    """Evaluate the SFT model on ARC multiple-choice benchmarks."""

    def __init__(
        self,
        config: GPTConfig,
        model_path=GPT_SFT_MODEL_PATH,
        split: str = "test",
        generation_max_length: int = 16,
        arc_configs: Iterable[str] | None = None,
    ) -> None:
        self.config = config
        self.model_path = model_path
        self.split = split
        self.max_length = generation_max_length
        self.arc_configs = tuple(arc_configs or ("ARC-Challenge", "ARC-Easy"))
        self.console = Console()

        self.inference = GPTInference(config=self.config, model_path=self.model_path)
        self.datasets = self._load_datasets(split=self.split)
        self._last_split_results: dict[str, ARCEvalResult] = {}

    def _load_datasets(self, split: str) -> dict[str, Dataset]:
        loaded: dict[str, Dataset] = {}
        for config in self.arc_configs:
            dataset = load_dataset("allenai/ai2_arc", config, split=split, trust_remote_code=False)
            if isinstance(dataset, Dataset):
                loaded[config] = dataset
            elif isinstance(dataset, dict) and "train" in dataset:
                loaded[config] = dataset[split]
        if not loaded:
            raise ValueError("No ARC datasets could be loaded")
        return loaded

    def _format_instruction(self, row: dict[str, str | dict]) -> str:
        question_text = str(row.get("question") or "").strip()
        choices: dict[str, list[str]] = row.get("choices") or {}  # type: ignore[assignment]
        labels: list[str] = choices.get("label") or []
        texts: list[str] = choices.get("text") or []
        choice_lines = [f"{label}. {text}" for label, text in zip(labels, texts, strict=False) if label and text]
        choice_block = "Choices:\n" + "\n".join(choice_lines) if choice_lines else ""
        instruction_parts = [
            "Answer the multiple-choice question with exactly one letter.",
        ]
        if question_text:
            instruction_parts.append(f"Question: {question_text}")
        if choice_block:
            instruction_parts.append(choice_block)
        return "\n\n".join(instruction_parts)

    def _build_prompt(self, row: dict[str, str | dict]) -> str:
        instruction = self._format_instruction(row)
        return f"Human: {instruction}\n\nAssistant:"

    def _extract_choice_letter(self, generated: str, prompt: str, valid_labels: list[str]) -> str:
        valid = [label.upper() for label in valid_labels if label]
        if not valid:
            return ""
        pattern = "|".join(re.escape(label) for label in sorted(valid, key=len, reverse=True))
        completion = generated[len(prompt) :].lstrip() if generated.startswith(prompt) else generated.strip()
        matches = list(re.finditer(rf"({pattern})", completion, re.IGNORECASE))
        if not matches:
            return ""
        return matches[-1].group(1).upper()

    def evaluate(self, max_samples: int | None = None, show_details: bool = False) -> ARCEvalResult:
        overall_correct = 0
        overall_total = 0
        self._last_split_results = {}
        for config_name, dataset in self.datasets.items():
            ds = dataset
            if max_samples is not None:
                limit = min(max_samples, len(ds))
                ds = ds.select(range(limit))
            split_correct = 0
            split_total = len(ds)
            table = None
            if show_details:
                table = Table(title=f"{config_name} Predictions")
                table.add_column("Question ID", justify="left")
                table.add_column("Target")
                table.add_column("Prediction")

            for row in ds:
                prompt = self._build_prompt(row)
                choices = row.get("choices") or {}
                labels = choices.get("label") or []
                generated = self.inference.generate_text(prompt, max_length=self.max_length)
                predicted = self._extract_choice_letter(generated, prompt, list(labels))
                answer = str(row.get("answerKey") or "").strip().upper()
                target_display = answer
                if predicted == answer:
                    split_correct += 1
                if table is not None:
                    table.add_row(str(row.get("id")), target_display or answer, predicted or "?")

            if table is not None:
                self.console.print(table)

            accuracy = split_correct / split_total if split_total > 0 else 0.0
            result = ARCEvalResult(accuracy=accuracy, total=split_total, correct=split_correct)
            self._last_split_results[config_name] = result
            self.console.print(f"[bold]{config_name}[/bold]: {result.accuracy:.2%} ({result.correct}/{result.total})")
            overall_correct += split_correct
            overall_total += split_total

        overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0
        overall_result = ARCEvalResult(accuracy=overall_accuracy, total=overall_total, correct=overall_correct)
        self.console.print(
            f"[bold]Overall[/bold]: {overall_result.accuracy:.2%} ({overall_result.correct}/{overall_result.total})"
        )
        return overall_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GPT SFT model on ARC datasets")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples per split")
    parser.add_argument(
        "--details",
        action="store_true",
        help="Show per-question prediction details",
    )
    args = parser.parse_args()

    cfg = GPTConfig()
    evaluator = ARCEval(config=cfg, split=args.split)
    evaluator.evaluate(max_samples=args.max_samples, show_details=args.details)
