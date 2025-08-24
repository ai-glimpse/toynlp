from evaluate import load
from tqdm import tqdm
from toynlp.transformer.config import TransformerConfig
from toynlp.transformer.inference import TransformerInference
from toynlp.transformer.dataset import get_dataset
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


class TransformerEvaluator:
    """Evaluation class for computing BLEU scores on transformer translation model."""

    def __init__(self, config: TransformerConfig, inference: TransformerInference | None = None) -> None:
        self.config = config
        self.inference = inference if inference is not None else TransformerInference(config)
        self.bleu_metric = load("bleu")
        self.dataset = get_dataset(
            dataset_path=self.config.dataset_path,
            dataset_name=self.config.dataset_name,
        )
        print(f"Loaded dataset with splits: {list(self.dataset.keys())}")

    def _normalize_text(self, text: str) -> str:
        text = text.strip()
        special_tokens = ["[BOS]", "[EOS]", "[PAD]", "[UNK]"]
        for token in special_tokens:
            text = text.replace(token, "")
        text = " ".join(text.split())
        text = text.replace(" .", ".")  # remove the space before punctuation
        return text

    def _prepare_references(self, references: list[str]) -> list[list[str]]:
        normalized_refs = [self._normalize_text(ref) for ref in references]
        return [[ref] for ref in normalized_refs]

    def _prepare_predictions(self, predictions: list[str]) -> list[str]:
        return [self._normalize_text(pred) for pred in predictions]

    def evaluate_split(
        self,
        split: str = "test",
        max_samples: int | None = None,
        batch_size: int | None = None,
    ) -> dict[str, float]:
        if max_samples is None:
            max_samples = self.config.evaluation_max_samples
        if batch_size is None:
            batch_size = self.config.evaluation_batch_size
        if split not in self.dataset:
            available_splits = list(self.dataset.keys())
            msg = f"Split '{split}' not found in dataset. Available: {available_splits}"
            raise ValueError(msg)
        print(f"Evaluating on {split} split...")
        split_data = self.dataset[split]
        total_samples = len(split_data)
        if max_samples is not None:
            total_samples = min(max_samples, total_samples)
            split_data = split_data.select(range(total_samples))
        print(f"Evaluating on {total_samples} samples...")
        source_texts = split_data[self.config.source_lang]
        target_texts = split_data[self.config.target_lang]
        predictions = []
        for i in tqdm(range(0, len(source_texts), batch_size), desc="Translating"):
            batch_sources = source_texts[i : i + batch_size]
            batch_predictions = self.inference.translate(batch_sources)
            predictions.extend(batch_predictions)
        references = self._prepare_references(target_texts)
        predictions = self._prepare_predictions(predictions)
        bleu_result = self.bleu_metric.compute(
            predictions=predictions,
            references=references,
        )
        results = {
            "bleu": bleu_result["bleu"],
            "bleu_1": bleu_result["precisions"][0],
            "bleu_2": bleu_result["precisions"][1],
            "bleu_3": bleu_result["precisions"][2],
            "bleu_4": bleu_result["precisions"][3],
            "brevity_penalty": bleu_result["brevity_penalty"],
            "length_ratio": bleu_result["length_ratio"],
            "num_samples": total_samples,
        }
        return results

    def evaluate_all_splits(
        self,
        max_samples_per_split: int | None = None,
        batch_size: int | None = None,
    ) -> dict[str, dict[str, float]]:
        results: dict[str, Any] = {}
        for split in self.dataset:
            try:
                split_results = self.evaluate_split(
                    split=split,
                    max_samples=max_samples_per_split,
                    batch_size=batch_size,
                )
                results[split] = split_results
                print(f"\n{split.upper()} Results:")
                print(f"  BLEU Score: {split_results['bleu']:.4f}")
                print(f"  BLEU-1: {split_results['bleu_1']:.4f}")
                print(f"  BLEU-2: {split_results['bleu_2']:.4f}")
                print(f"  BLEU-3: {split_results['bleu_3']:.4f}")
                print(f"  BLEU-4: {split_results['bleu_4']:.4f}")
                print(f"  Brevity Penalty: {split_results['brevity_penalty']:.4f}")
                print(f"  Length Ratio: {split_results['length_ratio']:.4f}")
                print(f"  Samples: {split_results['num_samples']}")
            except (RuntimeError, ValueError, KeyError) as e:
                print(f"Error evaluating {split} split: {e}")
                results[split] = {"error": str(e)}
        return results

    def compare_samples(
        self,
        split: str = "test",
        num_samples: int = 5,
        start_idx: int = 0,
    ) -> None:
        if split not in self.dataset:
            available_splits = list(self.dataset.keys())
            msg = f"Split '{split}' not found in dataset. Available: {available_splits}"
            raise ValueError(msg)
        split_data = self.dataset[split]
        end_idx = min(start_idx + num_samples, len(split_data))
        print(f"\n=== Sample Translations from {split} split ===")
        print(f"Showing samples {start_idx} to {end_idx - 1}")
        print("=" * 60)
        for i in range(start_idx, end_idx):
            source_text = split_data[i][self.config.source_lang]
            target_text = split_data[i][self.config.target_lang]
            prediction = self.inference.translate(source_text)
            print(f"\nSample {i + 1}:")
            print(f"Source ({self.config.source_lang}): {source_text}")
            print(f"Target ({self.config.target_lang}): {target_text}")
            print(f"Prediction: {prediction}")
            norm_target = self._normalize_text(target_text)
            norm_pred = self._normalize_text(prediction)
            sample_bleu = self.bleu_metric.compute(
                predictions=[norm_pred],
                references=[[norm_target]],
            )
            print(f"Sample BLEU: {sample_bleu['bleu']:.4f}")
            print("-" * 40)


def run_evaluation() -> None:
    print("Starting Transformer Model Evaluation...")
    config = TransformerConfig()
    evaluator = TransformerEvaluator(config)
    print("\n" + "=" * 60)
    print("SAMPLE TRANSLATIONS")
    print("=" * 60)
    evaluator.compare_samples(split="test", num_samples=3)
    print("\n" + "=" * 60)
    print("BLEU SCORE EVALUATION")
    print("=" * 60)
    results = {}
    print("Evaluating TRAIN split (limited to 200 samples)...")
    results["train"] = evaluator.evaluate_split(split="train", max_samples=200, batch_size=16)
    print("Evaluating VALIDATION split (FULL dataset)...")
    results["validation"] = evaluator.evaluate_split(split="validation", max_samples=None, batch_size=16)
    print("Evaluating TEST split (FULL dataset)...")
    results["test"] = evaluator.evaluate_split(split="test", max_samples=None, batch_size=16)
    for split, metrics in results.items():
        print(f"\n{split.upper()} Results:")
        print(f"  BLEU Score: {metrics['bleu']:.4f}")
        print(f"  BLEU-1: {metrics['bleu_1']:.4f}")
        print(f"  BLEU-2: {metrics['bleu_2']:.4f}")
        print(f"  BLEU-3: {metrics['bleu_3']:.4f}")
        print(f"  BLEU-4: {metrics['bleu_4']:.4f}")
        print(f"  Brevity Penalty: {metrics['brevity_penalty']:.4f}")
        print(f"  Length Ratio: {metrics['length_ratio']:.4f}")
        print(f"  Samples: {metrics['num_samples']}")
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for split, metrics in results.items():
        if "error" not in metrics:
            print(f"{split.upper()}: BLEU = {metrics['bleu']:.4f} ({metrics['num_samples']} samples)")
        else:
            print(f"{split.upper()}: Error - {metrics['error']}")


if __name__ == "__main__":
    run_evaluation()
