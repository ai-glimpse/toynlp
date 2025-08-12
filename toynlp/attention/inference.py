import torch
from pathlib import Path

from toynlp.attention.config import AttentionConfig
from toynlp.attention.model import Seq2SeqAttentionModel
from toynlp.attention.tokenizer import AttentionTokenizer
from toynlp.util import current_device
from toynlp.paths import ATTENTION_MODEL_PATH

import matplotlib.pyplot as plt  # type: ignore[unresolved-import]


class AttentionInference:
    """Attention model inference class for translation tasks."""

    def __init__(self, config: AttentionConfig, model_path: Path = ATTENTION_MODEL_PATH) -> None:
        """Initialize the inference class with model and tokenizers.

        Args:
            config: Attention configuration
            model_path: Path to the saved model file
        """
        self.config = config
        self.device = current_device

        # Load tokenizers
        self.source_tokenizer = AttentionTokenizer(lang=self.config.source_lang).load()
        self.target_tokenizer = AttentionTokenizer(lang=self.config.target_lang).load()

        # Load model
        self.model = Seq2SeqAttentionModel(self.config)
        if model_path.exists():
            # Try to load the complete model first, if it fails, load state_dict
            try:
                loaded_model = torch.load(model_path, map_location=self.device, weights_only=False)
                if isinstance(loaded_model, Seq2SeqAttentionModel):
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

    def translate(
        self,
        text: str,
        max_length: int | None = None,
        plot_attention: bool = False,
    ) -> str:
        """Translate text from source language to target language.

        Args:
            text: Input text to translate
            max_length: Maximum length of output sequence
            plot_attention: Whether to plot attention weights

        Returns:
            Translated text
        """
        if max_length is None:
            max_length = self.config.inference_max_length
        with torch.no_grad():
            # Preprocess input
            input_tensor = self.preprocess_text(text)

            # Get encoder outputs
            encoder_outputs, hidden = self.model.encoder(input_tensor)

            # Initialize decoder input with BOS token
            bos_token_id = self.target_tokenizer.token_to_id("[BOS]")
            eos_token_id = self.target_tokenizer.token_to_id("[EOS]")

            # Generate translation token by token
            output_tokens = []
            decoder_input_ids = torch.tensor([[bos_token_id]], dtype=torch.long).to(self.device)
            attention_weights = torch.zeros(max_length, input_tensor.shape[1])
            for i in range(max_length):
                context, attention_weight = self.model.attention(encoder_outputs, hidden)
                attention_weights[i] = attention_weight.squeeze(0).detach().cpu()
                # decoder output: (batch_size, 1, target_vocab_size)
                decoder_output, hidden = self.model.decoder(decoder_input_ids, context, hidden)
                # Forward through decoder
                # Get the token with highest probability
                next_token_id = decoder_output.argmax(dim=-1).squeeze().item()

                # Stop if we hit the EOS token
                if next_token_id == eos_token_id:
                    break

                output_tokens.append(next_token_id)

                # Use the predicted token as next input
                decoder_input_ids = torch.tensor([[next_token_id]], dtype=torch.long).to(self.device)

            # Convert tokens to text
            translation = self.postprocess_tokens(output_tokens)
            if plot_attention:
                self.plot_attention_weight(
                    attention_weights[: len(output_tokens)],
                    input_tensor.squeeze().tolist(),
                    output_tokens,
                )
            return translation

    def plot_attention_weight(
        self,
        attention_weight: torch.Tensor,
        source_token_ids: list[int],
        predict_token_ids: list[int],
    ) -> None:
        # Convert attention weights to numpy arrays
        attention_weights_np = attention_weight.cpu().numpy()
        # print(f"Attention weights shape: {attention_weights_np.shape}")
        source_tokens = self.source_tokenizer.decode(source_token_ids, skip_special_tokens=False).split()
        # print(f"source_tokens: {source_tokens}, source_token_ids: {source_token_ids}")
        predict_tokens = self.target_tokenizer.decode(predict_token_ids, skip_special_tokens=False).split()
        # print(f"predict_tokens: {predict_tokens}, predict_token_ids: {predict_token_ids}")

        # Plot each attention weight matrix
        plt.figure(figsize=(12, 12))
        plt.matshow(attention_weights_np, cmap="bone")
        plt.xlabel("Source Tokens")
        plt.ylabel("Target Tokens")
        plt.xticks(ticks=range(len(source_tokens)), labels=source_tokens, rotation=90)
        plt.yticks(ticks=range(len(predict_tokens)), labels=predict_tokens)
        # plt.show()
        plt.tight_layout()
        path = ATTENTION_MODEL_PATH.parent / "viz"
        path.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            f"{path}/attention_{' '.join(predict_tokens[1:5])}.png",
            dpi=300,
        )

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


def run_translation(config: AttentionConfig, plot_attention: bool = False) -> None:
    """Test function to demonstrate translation capabilities."""
    print("Loading Attention model for translation testing...")

    # Initialize inference
    inference = AttentionInference(config)

    # Test sentences (German to English)
    test_sentences = [
        "Ein Boston Terrier läuft über saftig-grünes Gras vor einem weißen Zaun.",
        "Ein Mann mit einem orangefarbenen Hut, der etwas anstarrt.",
    ]

    print(f"\nTranslating from {config.source_lang} to {config.target_lang}:")
    print("=" * 60)

    for i, sentence in enumerate(test_sentences, 1):
        translation = inference.translate(sentence, plot_attention=plot_attention)
        print(f"{i}. Source(input): {sentence}")
        print(f"   Target(output): {translation}")

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
    from toynlp.attention.config import create_config_from_cli

    # Create config from CLI
    config = create_config_from_cli()

    # Run translation test
    run_translation(config, plot_attention=True)
