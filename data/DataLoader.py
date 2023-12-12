import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import WikiText103, WikiText2
import numpy as np
import os
import sys
import json
from PIL import Image
import random


def get_train_dataset(sequence_length):
    train_iterator = WikiText2(split='train')
    tokenizer = tiktoken.get_encoding('p50k_base')
    # tokenizer = tiktoken.Encoding(
    #     name="my_encoding",
    #     pat_str=tokenizer._pat_str,
    #     mergeable_ranks=tokenizer._mergeable_ranks,
    #     special_tokens=
    # )
    data = [torch.tensor(tokenizer.encode(item)) for item in train_iterator]
    data = torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
    ds = TextDataset(data, sequence_length)
    return ds


class TextDataset(Dataset):
    def __init__(self, token_array, sequence_length):
        super().__init__()
        self.data = token_array
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) // self.sequence_length

    def __getitem__(self, idx):
        data = self.data[idx * self.sequence_length: (idx + 1) * self.sequence_length]
        targets = self.data[(idx * self.sequence_length + 1): ((idx + 1) * self.sequence_length) + 1]
        return data, targets


if __name__ == "__main__":
    ds = get_train_dataset(200)
    for idx, (data, targets) in enumerate(ds):
        print('here')
