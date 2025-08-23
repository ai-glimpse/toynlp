import torch
from pathlib import Path
from toynlp.transformer.config import TransformerConfig
from toynlp.transformer.model import TransformerModel
from toynlp.transformer.tokenizer import TransformerTokenizer
from toynlp.util import current_device
from toynlp.paths import TRANSFORMER_MODEL_PATH


class TransformerInference:
    """Transformer model inference class for translation tasks (greedy decoding)."""

    def __init__(self, config: TransformerConfig | None = None, model_path: Path = TRANSFORMER_MODEL_PATH) -> None:
        self.config = config if config is not None else TransformerConfig()
        self.device = current_device

        # Load tokenizer
        self.tokenizer = TransformerTokenizer().load()

        # Get padding_idx from tokenizer
        pad_token_id = self.tokenizer.token_to_id("[PAD]")

        # Load model
        self.model = TransformerModel(self.config, padding_idx=pad_token_id)
        if model_path.exists():
            try:
                loaded = torch.load(model_path, map_location=self.device, weights_only=False)
                if isinstance(loaded, TransformerModel):
                    self.model = loaded
                else:
                    self.model.load_state_dict(loaded)
                print(f"Model loaded from {model_path}")
            except (RuntimeError, TypeError, KeyError, FileNotFoundError) as e:
                print(f"Warning: Could not load model from {model_path}: {e}. Using untrained model.")
        else:
            print(f"Warning: Model file not found at {model_path}. Using untrained model.")
        self.model.to(self.device)
        self.model.eval()

    def preprocess_text(self, text: str) -> torch.Tensor:
        """Tokenize and convert input text to tensor."""
        # Tokenizer expects a string, returns Encoding object
        encoding = self.tokenizer.encode(text)
        token_ids = encoding.ids
        return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(self.device)

    def postprocess_tokens(self, token_ids: list[int]) -> str:
        """Convert token ids back to text."""
        text = self.tokenizer.decode(token_ids)
        # Remove special tokens and clean up
        text = text.replace("[BOS]", "").replace("[EOS]", "").replace("[PAD]", "").strip()
        return text

    def translate(self, text: str, max_length: int | None = None) -> str:
        """Translate text using greedy decoding."""
        if max_length is None:
            max_length = self.config.inference_max_length
        with torch.no_grad():
            input_tensor = self.preprocess_text(text)
            bos_token_id = self.tokenizer.token_to_id("[BOS]")
            eos_token_id = self.tokenizer.token_to_id("[EOS]")
            # Start with BOS
            output_tokens = [bos_token_id]
            for _ in range(max_length):
                tgt_tensor = torch.tensor([output_tokens], dtype=torch.long).to(self.device)
                logits = self.model(input_tensor, tgt_tensor)
                next_token_id = logits[0, -1].argmax(dim=-1).item()
                if next_token_id == eos_token_id:
                    break
                output_tokens.append(next_token_id)
            # Remove BOS for output
            return self.postprocess_tokens(output_tokens[1:])

    def translate_batch(self, texts: list[str], max_length: int | None = None) -> list[str]:
        """Translate a batch of texts using greedy decoding."""
        if max_length is None:
            max_length = self.config.inference_max_length
        return [self.translate(text, max_length) for text in texts]


def test_translation() -> None:
    """Test function to demonstrate transformer translation capabilities."""
    print("Loading Transformer model for translation testing...")
    config = TransformerConfig()
    inference = TransformerInference(config)
    test_sentences = [
        "Ein Mann sitzt auf einer Bank.",
        "Hallo, wie geht es dir?",
        "Ich liebe maschinelles Lernen.",
        "Das Wetter ist heute schön.",
        "Wo ist die nächste U-Bahn-Station?",
        "Kannst du mir helfen?",
    ]
    print(f"\nTranslating from {inference.config.source_lang} to {inference.config.target_lang}:")
    print("=" * 60)
    for i, sentence in enumerate(test_sentences, 1):
        try:
            translation = inference.translate(sentence)
            print(f"{i}. Source: {sentence}")
            print(f"   Target: {translation}")
            print()
        except (RuntimeError, ValueError, KeyError, FileNotFoundError) as e:
            print(f"{i}. Source: {sentence}")
            print(f"   Error: {e}")
            print()
    print("Testing batch translation...")
    try:
        batch_translations = inference.translate_batch(test_sentences[:3])
        print("Batch translation results:")
        for src, tgt in zip(test_sentences[:3], batch_translations, strict=True):
            print(f"  {src} -> {tgt}")
    except (RuntimeError, ValueError, KeyError, FileNotFoundError) as e:
        print(f"Batch translation error: {e}")


if __name__ == "__main__":
    test_translation()
