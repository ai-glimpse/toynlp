from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm
from typing import Any

from toynlp.fasttext.config import FastTextConfig, create_config_from_cli
from toynlp.fasttext.inference import load_model_and_tokenizer, classify_text
from toynlp.fasttext.dataset import get_dataset


class FastTextEvaluator:
    """Evaluation class for computing classification metrics on FastText model."""

    def __init__(self, config: FastTextConfig) -> None:
        """Initialize the evaluator.

        Args:
            config: FastText configuration
        """
        self.config = config

        # Load model and tokenizer
        try:
            self.model, self.tokenizer = load_model_and_tokenizer()
            print("✅ Model and tokenizer loaded successfully")
        except FileNotFoundError:
            print("❌ No trained model found. Please train a model first.")
            raise

        # Load dataset
        self.dataset = get_dataset(
            dataset_path=self.config.dataset_path,
            dataset_name=self.config.dataset_name,
        )
        print(f"✅ Loaded dataset with splits: {list(self.dataset.keys())}")

    def evaluate_split(
        self,
        split: str = "test",
        max_samples: int | None = None,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate model on a specific dataset split.

        Args:
            split: Dataset split to evaluate ('train', 'test', etc.)
            max_samples: Maximum number of samples to evaluate (None for all)
            batch_size: Batch size for inference (uses config default if None)

        Returns:
            Dictionary containing various classification metrics
        """
        if split not in self.dataset:
            available_splits = list(self.dataset.keys())
            msg = f"Split '{split}' not found. Available splits: {available_splits}"
            raise ValueError(msg)

        eval_dataset = self.dataset[split]

        if max_samples is not None:
            eval_dataset = eval_dataset.select(range(min(max_samples, len(eval_dataset))))

        if batch_size is None:
            batch_size = self.config.batch_size

        print(f"📊 Evaluating on {len(eval_dataset)} samples from '{split}' split")

        # Extract texts and labels
        texts = eval_dataset["text"]
        true_labels = eval_dataset["label"]

        # Get predictions
        predictions = []  # type: ignore[var-annotated]

        # Process in batches for memory efficiency
        for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating"):
            batch_texts = texts[i : i + batch_size]
            batch_predictions = classify_text(
                batch_texts,
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=self.config.max_length,
            )
            predictions.extend(batch_predictions)

        # Ensure predictions and labels have the same length
        assert len(predictions) == len(true_labels), "Mismatch in predictions and labels length"

        # Compute metrics
        metrics = self._compute_metrics(true_labels, predictions)

        # Add split info
        metrics["split"] = split
        metrics["num_samples"] = len(eval_dataset)

        return metrics

    def _compute_metrics(self, true_labels: list[int], predictions: list[int]) -> dict[str, Any]:
        """Compute classification metrics.

        Args:
            true_labels: Ground truth labels
            predictions: Predicted labels

        Returns:
            Dictionary containing various metrics
        """
        # Basic accuracy
        accuracy = accuracy_score(true_labels, predictions)

        # Precision, recall, F1 (macro and weighted averages)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            true_labels,
            predictions,
            average="macro",
            zero_division=0,
        )

        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            true_labels,
            predictions,
            average="weighted",
            zero_division=0,
        )

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            true_labels,
            predictions,
            average=None,
            zero_division=0,
        )

        # Confusion matrix
        conf_matrix = confusion_matrix(true_labels, predictions)

        # Classification report
        class_report = classification_report(
            true_labels,
            predictions,
            output_dict=True,
            zero_division=0,
        )

        # Organize metrics
        metrics = {
            "accuracy": float(accuracy),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(precision_weighted),
            "recall_weighted": float(recall_weighted),
            "f1_weighted": float(f1_weighted),
            "confusion_matrix": conf_matrix.tolist(),
            "per_class_metrics": {
                "precision": precision_per_class.tolist(),
                "recall": recall_per_class.tolist(),
                "f1": f1_per_class.tolist(),
                "support": support_per_class.tolist(),
            },
            "classification_report": class_report,
        }

        return metrics

    def print_metrics(self, metrics: dict[str, Any]) -> None:
        """Print evaluation metrics in a formatted way.

        Args:
            metrics: Dictionary containing evaluation metrics
        """
        print("\n" + "=" * 80)
        print("FASTTEXT EVALUATION RESULTS")
        print("=" * 80)
        print(f"Dataset Split: {metrics.get('split', 'unknown')}")
        print(f"Number of Samples: {metrics.get('num_samples', 'unknown')}")
        print("-" * 80)

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score (Macro): {metrics['f1_macro']:.4f}")
        print(f"F1 Score (Weighted): {metrics['f1_weighted']:.4f}")
        print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
        print(f"Recall (Macro): {metrics['recall_macro']:.4f}")

        print("\n" + "-" * 40)
        print("PER-CLASS METRICS")
        print("-" * 40)

        per_class = metrics["per_class_metrics"]
        num_classes = len(per_class["precision"])

        print(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 60)

        for i in range(num_classes):
            precision = per_class["precision"][i]
            recall = per_class["recall"][i]
            f1 = per_class["f1"][i]
            support = per_class["support"][i]
            print(f"{i:<8} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<10}")

        print("\n" + "-" * 40)
        print("CONFUSION MATRIX")
        print("-" * 40)

        conf_matrix = metrics["confusion_matrix"]
        num_classes = len(conf_matrix)

        # Print header
        print("True\\Pred", end="")
        for i in range(num_classes):
            print(f"{i:>8}", end="")
        print()

        # Print matrix
        for i, row in enumerate(conf_matrix):
            print(f"{i:>8}", end="")
            for val in row:
                print(f"{val:>8}", end="")
            print()

        print("=" * 80)

    def evaluate_all_splits(self, max_samples: int | None = None) -> dict[str, dict[str, Any]]:
        """Evaluate model on all available dataset splits.

        Args:
            max_samples: Maximum number of samples per split (None for all)

        Returns:
            Dictionary mapping split names to their evaluation metrics
        """
        all_metrics = {}

        for split in ["train", "test"]:
            print(f"\n🔍 Evaluating split: {split}")
            try:
                metrics = self.evaluate_split(split, max_samples)
                all_metrics[split] = metrics
                self.print_metrics(metrics)
            except (ValueError, FileNotFoundError, RuntimeError) as e:
                print(f"❌ Error evaluating {split}: {e}")
                continue

        return all_metrics

    def quick_evaluation(self, num_samples: int = 100) -> None:
        """Perform a quick evaluation on a small subset of test data.

        Args:
            num_samples: Number of samples to evaluate
        """
        print(f"🚀 Quick evaluation on {num_samples} samples")

        try:
            metrics = self.evaluate_split("test", max_samples=num_samples)
            self.print_metrics(metrics)
        except ValueError:
            # Fallback to first available split if test doesn't exist
            available_splits = list(self.dataset.keys())
            if available_splits:
                split = available_splits[0]
                print(f"Test split not found, using '{split}' instead")
                metrics = self.evaluate_split(split, max_samples=num_samples)
                self.print_metrics(metrics)
            else:
                print("❌ No data splits available for evaluation")


def main() -> None:
    """Main function for running FastText evaluation."""
    # Load configuration
    config = create_config_from_cli()

    print("=" * 80)
    print("FASTTEXT MODEL EVALUATION")
    print("=" * 80)
    print(f"Dataset: {config.dataset_path}")
    print("Model Path: From configuration")
    print("=" * 80)

    try:
        # Create evaluator
        evaluator = FastTextEvaluator(config)

        # Perform evaluation
        print("\n🔍 Starting evaluation...")
        evaluator.evaluate_all_splits()

    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"❌ Evaluation failed: {e}")
        return

    print("\n✅ Evaluation completed!")


if __name__ == "__main__":
    main()
