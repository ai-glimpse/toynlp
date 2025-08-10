import torch
from pathlib import Path

from toynlp.seq2seq.config import Seq2SeqConfig
from toynlp.seq2seq.model import Seq2SeqModel
from toynlp.seq2seq.tokenizer import Seq2SeqTokenizer
from toynlp.util import current_device
from toynlp.paths import SEQ2SEQ_MODEL_PATH


class Seq2SeqInference:
    """Seq2Seq model inference class for translation tasks."""

    def __init__(self, config: Seq2SeqConfig, model_path: Path = SEQ2SEQ_MODEL_PATH) -> None:
        """Initialize the inference class with model and tokenizers.

        Args:
            config: Configuration object
            model_path: Path to the saved model file
        """
        self.config = config
        self.device = current_device

        # Load tokenizers
        self.source_tokenizer = Seq2SeqTokenizer(lang=self.config.source_lang).load()
        self.target_tokenizer = Seq2SeqTokenizer(lang=self.config.target_lang).load()

        # Load model
        self.model = Seq2SeqModel(self.config)
        if model_path.exists():
            # Try to load the complete model first, if it fails, load state_dict
            try:
                loaded_model = torch.load(model_path, map_location=self.device, weights_only=False)
                if isinstance(loaded_model, Seq2SeqModel):
                    self.model = loaded_model
                else:
                    self.model.load_state_dict(loaded_model)
                print(f"Model loaded from {model_path}")
            except (RuntimeError, TypeError, KeyError, FileNotFoundError) as e:
                print(f"Warning: Could not load model from {model_path}: {e}. Using untrained model.")
        else:
            print(f"Warning: Model file not found at {model_path}. Using untrained model.")

        self.model.to(self.device)
        self.model.eval()

    def preprocess_text(self, text: str) -> torch.Tensor:
        """Preprocess input text and convert to tensor.

        Args:
            text: Input text to preprocess

        Returns:
            Tensor of token ids
        """
        # Encode the text using source tokenizer
        token_ids = self.source_tokenizer.encode(text).ids
        # Convert to tensor and add batch dimension
        return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(self.device)

    def postprocess_tokens(self, token_ids: list[int]) -> str:
        """Convert token ids back to text.

        Args:
            token_ids: List of token ids

        Returns:
            Decoded text string
        """
        # Decode using target tokenizer
        text = self.target_tokenizer.decode(token_ids)
        # Remove special tokens and clean up
        text = text.replace("[BOS]", "").replace("[EOS]", "").replace("[PAD]", "").strip()
        return text

    def translate(self, text: str, max_length: int | None = None) -> str:
        """Translate text from source language to target language.

        Args:
            text: Input text to translate
            max_length: Maximum length of output sequence

        Returns:
            Translated text
        """
        if max_length is None:
            max_length = self.config.inference_max_length
        with torch.no_grad():
            # Preprocess input
            input_tensor = self.preprocess_text(text)

            # Get encoder outputs
            hidden, cell = self.model.encoder(input_tensor)

            # Initialize decoder input with BOS token
            bos_token_id = self.target_tokenizer.token_to_id("[BOS]")
            eos_token_id = self.target_tokenizer.token_to_id("[EOS]")

            decoder_input = torch.tensor([[bos_token_id]], dtype=torch.long).to(self.device)

            # Generate translation token by token
            output_tokens = []

            for _ in range(max_length):
                # Forward through decoder
                decoder_output, hidden, cell = self.model.decoder(decoder_input, hidden, cell)

                # Get the token with highest probability
                next_token_id = decoder_output.argmax(dim=-1).squeeze().item()

                # Stop if we hit the EOS token
                if next_token_id == eos_token_id:
                    break

                output_tokens.append(next_token_id)

                # Use the predicted token as next input
                decoder_input = torch.tensor([[next_token_id]], dtype=torch.long).to(self.device)

            # Convert tokens to text
            translation = self.postprocess_tokens(output_tokens)
            return translation

    def translate_batch(self, texts: list[str], max_length: int | None = None) -> list[str]:
        """Translate a batch of texts.

        Args:
            texts: List of input texts to translate
            max_length: Maximum length of output sequences

        Returns:
            List of translated texts
        """
        if max_length is None:
            max_length = self.config.inference_max_length
        translations = []
        for text in texts:
            translation = self.translate(text, max_length)
            translations.append(translation)
        return translations


def test_translation() -> None:
    """Test function to demonstrate translation capabilities."""
    print("Loading Seq2Seq model for translation testing...")

    # Initialize inference with default config
    config = Seq2SeqConfig()
    inference = Seq2SeqInference(config)

    # Test sentences (German to English)
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
        except (RuntimeError, ValueError, KeyError) as e:
            print(f"{i}. Source: {sentence}")
            print(f"   Error: {e}")
            print()

    # Test batch translation
    print("Testing batch translation...")
    try:
        batch_translations = inference.translate_batch(test_sentences[:3])
        print("Batch translation results:")
        for src, tgt in zip(test_sentences[:3], batch_translations, strict=True):
            print(f"  {src} -> {tgt}")
    except (RuntimeError, ValueError, KeyError) as e:
        print(f"Batch translation error: {e}")


if __name__ == "__main__":
    test_translation()
