from toynlp.gpt.config import GPTConfig
from toynlp.gpt.model import GPTModel
from toynlp.gpt.tokenizer import GPTTokenizer

from toynlp.paths import GPT_MODEL_PATH
import torch
from pathlib import Path
from toynlp.util import current_device


class GPTInference:
    """GPT model inference class for text generation tasks."""

    def __init__(self, config: GPTConfig, model_path: Path = GPT_MODEL_PATH) -> None:
        """Initialize the inference class with model and tokenizer.

        Args:
            config: GPT configuration
            model_path: Path to the saved model file
        """
        self.config = config
        self.device = current_device

        # Load tokenizer
        self.gpt_tokenizer = GPTTokenizer().load()

        # Load model
        self.model = GPTModel(self.config, padding_idx=self.gpt_tokenizer.token_to_id("<pad>"))
        if model_path.exists():
            # Try to load the complete model first, if it fails, load state_dict
            try:
                loaded_model = torch.load(model_path, map_location=self.device, weights_only=False)
                if isinstance(loaded_model, GPTModel):
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
            Preprocessed input tensor
        """
        # Tokenize text
        input_ids = self.gpt_tokenizer.encode(text).ids
        # Convert to tensor
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        return input_tensor

    def generate_text(self, prompt: str, max_length: int = 50) -> str:
        """Generate text based on the input prompt.

        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text

        Returns:
            Generated text
        """
        input_tensor = self.preprocess_text(prompt)
        generated_ids = input_tensor

        with torch.no_grad():
            length = 0
            while length < max_length:
                outputs = self.model(generated_ids)
                next_token_logits = outputs[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
                generated_ids = torch.cat((generated_ids, next_token_id), dim=1)
                if next_token_id.item() == self.gpt_tokenizer.token_to_id("."):
                    break
                length += 1

        generated_text = self.gpt_tokenizer.decode(generated_ids.squeeze().tolist())
        return generated_text


if __name__ == "__main__":
    # Example usage of GPTInference
    config = GPTConfig()
    gpt_inference = GPTInference(config)

    prompt = "This is"
    generated_text = gpt_inference.generate_text(prompt, max_length=100)
    print("Generated Text:")
    print(generated_text)
