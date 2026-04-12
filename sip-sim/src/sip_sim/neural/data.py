"""Shared data utilities for neural experiments."""

import tiktoken


def get_tokenizer():
    """GPT-2 BPE tokenizer via tiktoken."""
    return tiktoken.get_encoding("gpt2")
