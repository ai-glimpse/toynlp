import torch
import random


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


current_device = get_device()


def setup_seed(seed: int) -> None:
    random.seed(seed)
    print(f"Setting random seed to {seed}")
    torch.manual_seed(seed)
    print(f"Setting torch seed to {seed}")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"Setting CUDA seed to {seed}")


def set_deterministic_mode() -> None:
    """Set PyTorch to deterministic mode for reproducibility."""
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Running in CPU mode.")
        return
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("Set PyTorch to deterministic mode")
