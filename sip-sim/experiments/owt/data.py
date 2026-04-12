"""OpenWebText dataset loading with tiktoken tokenizer.

OpenWebText: ~8GB, ~9B tokens. Open recreation of GPT-2's training data.
Standard benchmark for reproducing GPT-2 results.
"""

import os

import torch
from torch.utils.data import Dataset, DataLoader

from sip_sim.neural import get_tokenizer


class OpenWebTextDataset(Dataset):
    """Tokenised OpenWebText, packed into fixed-length chunks."""

    def __init__(self, split="train", context_length=1024, data_dir=None):
        self.context_length = context_length
        self.enc = get_tokenizer()

        cache_path = self._cache_path(split, data_dir)
        if os.path.exists(cache_path):
            print(f"Loading cached tokens from {cache_path}")
            self.tokens = torch.load(cache_path, weights_only=True)
        else:
            print(f"Tokenising OpenWebText {split} split...")
            self.tokens = self._tokenise(split)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(self.tokens, cache_path)
            print(f"Cached {len(self.tokens)} tokens to {cache_path}")

        self.n_examples = (len(self.tokens) - 1) // context_length
        print(f"  {split}: {len(self.tokens)/1e9:.2f}B tokens, "
              f"{self.n_examples} examples at context_length={context_length}")

    def _cache_path(self, split, data_dir):
        base = data_dir or os.path.join(os.path.dirname(__file__), ".data")
        return os.path.join(base, f"openwebtext_{split}_tokens.pt")

    def _tokenise(self, split):
        from datasets import load_dataset

        ds = load_dataset("openwebtext", split="train")
        ds = ds.train_test_split(test_size=0.005, seed=42)

        if split == "train":
            ds_split = ds["train"]
        else:
            ds_split = ds["test"]

        all_tokens = []
        eot = self.enc.eot_token
        for i, example in enumerate(ds_split):
            text = example["text"]
            tokens = self.enc.encode_ordinary(text)
            all_tokens.extend(tokens)
            all_tokens.append(eot)
            if (i + 1) % 500_000 == 0:
                print(f"  tokenised {i+1} examples, {len(all_tokens)/1e9:.2f}B tokens")

        return torch.tensor(all_tokens, dtype=torch.long)

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        start = idx * self.context_length
        end = start + self.context_length + 1
        chunk = self.tokens[start:end]
        return chunk[:-1], chunk[1:]


def get_dataloaders(batch_size=8, context_length=1024, num_workers=4, data_dir=None):
    """Create train and validation dataloaders."""
    train_ds = OpenWebTextDataset("train", context_length, data_dir)
    val_ds = OpenWebTextDataset("validation", context_length, data_dir)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=True)

    return train_loader, val_loader
