from datasets import load_dataset
from torch.utils.data import DataLoader


# load dataset
def load_text_dataset():
    """
    https://huggingface.co/datasets/Salesforce/wikitext
    https://huggingface.co/docs/transformers/perplexity#example-calculating-perplexity-with-gpt-2-in--transformers
    """
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-v1")
    dataloader = DataLoader(ds['train'], batch_size=32)
    return dataloader

