import torch
from torch.nn import functional
from tokenizers import Tokenizer

from toynlp.util import current_device
from toynlp.paths import FASTTEXT_MODEL_PATH
from toynlp.fasttext.model import FastTextModel
from toynlp.fasttext.tokenizer import FastTextTokenizer


def load_model_and_tokenizer() -> tuple[FastTextModel, Tokenizer]:
    """Load the trained FastText model and tokenizer.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer
    """
    # Load tokenizer
    tokenizer = FastTextTokenizer().load()

    # Load model
    model = torch.load(str(FASTTEXT_MODEL_PATH), weights_only=False)
    model.to(current_device)
    model.eval()

    return model, tokenizer


def preprocess_text(
    text: str | list[str],
    tokenizer: Tokenizer,
    max_length: int = 2500,
) -> torch.Tensor:
    """Preprocess text for inference.

    Args:
        text: Single text string or list of text strings
        tokenizer: The FastText tokenizer
        max_length: Maximum sequence length

    Returns:
        torch.Tensor: Preprocessed input tensor
    """
    if isinstance(text, str):
        text = [text]

    batch_input = []
    pad_id = tokenizer.token_to_id("[PAD]")

    for single_text in text:
        token_ids = tokenizer.encode(single_text).ids
        # Truncate if too long
        token_ids = token_ids[:max_length]
        batch_input.append(torch.tensor(token_ids, dtype=torch.long))

    # Pad sequences to same length
    max_len = max(len(seq) for seq in batch_input)
    padded_batch = []
    for seq in batch_input:
        if len(seq) < max_len:
            padding = torch.full((max_len - len(seq),), pad_id, dtype=torch.long)
            padded_seq = torch.cat([seq, padding])
        else:
            padded_seq = seq
        padded_batch.append(padded_seq)

    return torch.stack(padded_batch).to(current_device)


def classify_text(
    text: str | list[str],
    model: FastTextModel | None = None,
    tokenizer: Tokenizer | None = None,
    return_probabilities: bool = False,
    max_length: int = 2500,
) -> list[int] | tuple[list[int], list[list[float]]]:
    """Classify text using the trained FastText model.

    Args:
        text: Single text string or list of text strings to classify
        model: Pre-loaded model (optional, will load if not provided)
        tokenizer: Pre-loaded tokenizer (optional, will load if not provided)
        return_probabilities: Whether to return class probabilities
        max_length: Maximum sequence length

    Returns:
        If return_probabilities is False: List of predicted class indices
        If return_probabilities is True: Tuple of (predictions, probabilities)
    """
    # Load model and tokenizer if not provided
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer()

    # Preprocess input
    input_tensor = preprocess_text(text, tokenizer, max_length)

    # Get predictions
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = functional.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)

    # Convert to lists
    predictions_list = predictions.cpu().tolist()
    probabilities_list = probabilities.cpu().tolist()

    if return_probabilities:
        return predictions_list, probabilities_list
    return predictions_list


def predict_single_text(
    text: str,
    model: FastTextModel | None = None,
    tokenizer: Tokenizer | None = None,
    class_names: list[str] | None = None,
) -> dict:
    """Predict a single text with detailed output.

    Args:
        text: Text string to classify
        model: Pre-loaded model (optional)
        tokenizer: Pre-loaded tokenizer (optional)
        class_names: List of class names for mapping (optional)

    Returns:
        dict: Dictionary containing prediction details
    """
    result = classify_text(
        text, model, tokenizer, return_probabilities=True,
    )
    # When return_probabilities=True, we get a tuple
    assert isinstance(result, tuple)
    predictions, probabilities = result

    prediction = predictions[0]
    probs = probabilities[0]

    result = {
        "text": text,
        "predicted_class": prediction,
        "confidence": max(probs),
        "probabilities": probs,
    }

    if class_names is not None:
        result["predicted_class_name"] = class_names[prediction]
        result["class_probabilities"] = {
            class_names[i]: prob for i, prob in enumerate(probs)
        }

    return result


def batch_inference(
    texts: list[str],
    batch_size: int = 32,
    model: FastTextModel | None = None,
    tokenizer: Tokenizer | None = None,
    max_length: int = 2500,
) -> list[int]:
    """Perform batch inference on a list of texts.

    Args:
        texts: List of text strings to classify
        batch_size: Batch size for processing
        model: Pre-loaded model (optional)
        tokenizer: Pre-loaded tokenizer (optional)
        max_length: Maximum sequence length

    Returns:
        list: List of predicted class indices
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_model_and_tokenizer()

    all_predictions = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_predictions = classify_text(
            batch_texts, model, tokenizer, max_length=max_length,
        )
        all_predictions.extend(batch_predictions)

    return all_predictions


def evaluate_sample_texts() -> None:
    """Evaluate some sample texts for demonstration."""
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Sample texts (assuming binary sentiment classification)
    sample_texts = [
        "This movie was absolutely amazing! I loved every minute of it.",
        "Terrible film, waste of time. Acting was awful.",
        "It was okay, nothing special but not bad either.",
        "One of the best movies I've ever seen! Highly recommended.",
        "Boring and predictable. Could barely stay awake.",
    ]

    class_names = ["negative", "positive"]  # Assuming binary classification

    print("FastText Classification Results:")
    print("=" * 50)

    for text in sample_texts:
        result = predict_single_text(text, model, tokenizer, class_names)
        print(f"Text: {text}")
        print(f"Prediction: {result['predicted_class_name']} (confidence: {result['confidence']:.3f})")
        print(f"Probabilities: {result['class_probabilities']}")
        print("-" * 50)


if __name__ == "__main__":
    evaluate_sample_texts()
