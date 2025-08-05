from evaluate import load
from tqdm import tqdm

from toynlp.seq2seq.config import Seq2SeqConfig
from toynlp.seq2seq.inference import Seq2SeqInference
from toynlp.seq2seq.dataset import get_dataset


class Seq2SeqEvaluator:
    """Evaluation class for computing BLEU scores on seq2seq translation model."""

    def __init__(self, inference: Seq2SeqInference | None = None) -> None:
        """Initialize the evaluator.

        Args:
            inference: Seq2SeqInference instance. If None, creates a new one.
        """
        self.config = Seq2SeqConfig()
        self.inference = inference if inference is not None else Seq2SeqInference()

        # Load BLEU metric
        self.bleu_metric = load("bleu")

        # Load dataset
        self.dataset = get_dataset(
            dataset_path=self.config.dataset.path,
            dataset_name=self.config.dataset.name,
        )

        print(f"Loaded dataset with splits: {list(self.dataset.keys())}")

    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent BLEU computation.

        Args:
            text: Input text to normalize

        Returns:
            Normalized text
        """
        # Remove extra whitespace and convert to lowercase
        text = text.strip().lower()

        # Remove special tokens that might be left in translations
        special_tokens = ["[BOS]", "[EOS]", "[PAD]", "[UNK]"]
        for token in special_tokens:
            text = text.replace(token.lower(), "")

        # Clean up multiple spaces
        text = " ".join(text.split())

        return text

    def _prepare_references(self, references: list[str]) -> list[list[str]]:
        """Prepare reference texts for BLEU computation.

        Args:
            references: List of reference sentences

        Returns:
            List of lists, where each inner list contains one reference
        """
        normalized_refs = [self._normalize_text(ref) for ref in references]
        # BLEU expects references as list of lists (multiple references per sentence)
        return [[ref] for ref in normalized_refs]

    def _prepare_predictions(self, predictions: list[str]) -> list[str]:
        """Prepare prediction texts for BLEU computation.

        Args:
            predictions: List of prediction sentences

        Returns:
            List of normalized predictions
        """
        return [self._normalize_text(pred) for pred in predictions]

    def evaluate_split(
        self,
        split: str = "test",
        max_samples: int | None = None,
        batch_size: int = 32,
    ) -> dict[str, float]:
        """Evaluate model on a specific dataset split.

        Args:
            split: Dataset split to evaluate ('train', 'validation', 'test')
            max_samples: Maximum number of samples to evaluate (None for all)
            batch_size: Batch size for translation

        Returns:
            Dictionary containing BLEU scores and other metrics
        """
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

        # Extract source and target texts
        source_texts = split_data[self.config.dataset.source]
        target_texts = split_data[self.config.dataset.target]

        # Generate translations in batches
        predictions = []

        for i in tqdm(range(0, len(source_texts), batch_size), desc="Translating"):
            batch_sources = source_texts[i:i + batch_size]
            batch_predictions = self.inference.translate_batch(batch_sources)
            predictions.extend(batch_predictions)

        # Prepare texts for BLEU computation
        references = self._prepare_references(target_texts)
        predictions = self._prepare_predictions(predictions)

        # Compute BLEU scores
        bleu_result = self.bleu_metric.compute(
            predictions=predictions,
            references=references,
        )

        # Additional metrics
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
        batch_size: int = 32,
    ) -> dict[str, dict[str, float]]:
        """Evaluate model on all available dataset splits.

        Args:
            max_samples_per_split: Maximum samples per split (None for all)
            batch_size: Batch size for translation

        Returns:
            Dictionary with results for each split
        """
        results = {}

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
        """Display sample translations for manual inspection.

        Args:
            split: Dataset split to sample from
            num_samples: Number of samples to display
            start_idx: Starting index for sampling
        """
        if split not in self.dataset:
            available_splits = list(self.dataset.keys())
            msg = f"Split '{split}' not found in dataset. Available: {available_splits}"
            raise ValueError(msg)

        split_data = self.dataset[split]
        end_idx = min(start_idx + num_samples, len(split_data))

        print(f"\n=== Sample Translations from {split} split ===")
        print(f"Showing samples {start_idx} to {end_idx-1}")
        print("=" * 60)

        for i in range(start_idx, end_idx):
            source_text = split_data[i][self.config.dataset.source]
            target_text = split_data[i][self.config.dataset.target]
            prediction = self.inference.translate(source_text)

            print(f"\nSample {i+1}:")
            print(f"Source ({self.config.dataset.source}): {source_text}")
            print(f"Target ({self.config.dataset.target}): {target_text}")
            print(f"Prediction: {prediction}")

            # Compute BLEU for this single sample
            norm_target = self._normalize_text(target_text)
            norm_pred = self._normalize_text(prediction)

            sample_bleu = self.bleu_metric.compute(
                predictions=[norm_pred],
                references=[[norm_target]],
            )
            print(f"Sample BLEU: {sample_bleu['bleu']:.4f}")
            print("-" * 40)


def run_evaluation() -> None:
    """Run comprehensive evaluation on the seq2seq model."""
    print("Starting Seq2Seq Model Evaluation...")

    # Initialize evaluator
    evaluator = Seq2SeqEvaluator()

    # Show some sample translations first
    print("\n" + "="*60)
    print("SAMPLE TRANSLATIONS")
    print("="*60)
    evaluator.compare_samples(split="test", num_samples=3)

    # Evaluate on all splits with full datasets
    print("\n" + "="*60)
    print("BLEU SCORE EVALUATION")
    print("="*60)

    # Full evaluation on validation and test splits
    # Keep train split limited to avoid overly long evaluation
    results = {}

    # Train split - limited to 200 samples to save time
    print("Evaluating TRAIN split (limited to 200 samples)...")
    results["train"] = evaluator.evaluate_split(split="train", max_samples=200, batch_size=16)

    # Validation split - FULL dataset
    print("Evaluating VALIDATION split (FULL dataset)...")
    results["validation"] = evaluator.evaluate_split(split="validation", max_samples=None, batch_size=16)

    # Test split - FULL dataset
    print("Evaluating TEST split (FULL dataset)...")
    results["test"] = evaluator.evaluate_split(split="test", max_samples=None, batch_size=16)

    # Print results for each split
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

    # Summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    for split, metrics in results.items():
        if "error" not in metrics:
            print(f"{split.upper()}: BLEU = {metrics['bleu']:.4f} ({metrics['num_samples']} samples)")
        else:
            print(f"{split.upper()}: Error - {metrics['error']}")


if __name__ == "__main__":
    run_evaluation()
