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

    def preprocess_texts(self, texts: list[str]) -> torch.Tensor:
        """Tokenize and convert a batch of input texts to a tensor."""
        tokenized_texts = [self.tokenizer.encode(text).ids for text in texts]
        max_length = max(len(tokens) for tokens in tokenized_texts)
        padded_tokens = [
            tokens + [self.tokenizer.token_to_id("[PAD]")] * (max_length - len(tokens))
            for tokens in tokenized_texts
        ]
        return torch.tensor(padded_tokens, dtype=torch.long).to(self.device)

    def postprocess_batch_tokens(self, batch_token_ids: list[list[int]]) -> list[str]:
        """Convert a batch of token ids back to text."""
        texts = []
        for token_ids in batch_token_ids:
            text = self.tokenizer.decode(token_ids)
            text = text.replace("[BOS]", "").replace("[EOS]", "").replace("[PAD]", "").replace(" .", ".").strip()
            texts.append(text)
        return texts

    def translate(self, texts: list[str], max_length: int | None = None) -> list[str]:
        """Translate text(s) using greedy decoding with early stopping for completed sequences."""
        if max_length is None:
            max_length = self.config.inference_max_length

        with torch.no_grad():
            input_tensor = self.preprocess_texts(texts)
            batch_size = input_tensor.size(0)
            bos_token_id = self.tokenizer.token_to_id("[BOS]")
            eos_token_id = self.tokenizer.token_to_id("[EOS]")

            # Initialize output tokens with BOS for each sequence in the batch
            output_tokens = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=self.device)

            # Track which sequences are still active (not finished)
            active_sequences = torch.ones(batch_size, dtype=torch.bool, device=self.device)

            for _ in range(max_length):
                logits = self.model(input_tensor, output_tokens)
                next_token_ids = logits[:, -1, :].argmax(dim=-1)

                # Update output tokens
                next_token_ids = next_token_ids.unsqueeze(1)
                output_tokens = torch.cat([output_tokens, next_token_ids], dim=1)

                # Mark sequences as finished if EOS is generated
                active_sequences &= ~(next_token_ids.squeeze() == eos_token_id)

                # Stop decoding if all sequences are finished
                if not active_sequences.any():
                    break

            # Remove BOS and truncate at EOS for each sequence
            batch_output_tokens = []
            for token_seq in output_tokens.tolist():
                truncated_tokens = (
                    token_seq[1:token_seq.index(eos_token_id)]
                    if eos_token_id in token_seq
                    else token_seq[1:]
                )
                batch_output_tokens.append(truncated_tokens)

            results = self.postprocess_batch_tokens(batch_output_tokens)
        return results


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
    batch_translations = inference.translate(test_sentences)
    print("Batch translation results:")
    for src, tgt in zip(test_sentences, batch_translations, strict=True):
        print(f"  {src} -> {tgt}")


if __name__ == "__main__":
    test_translation()
