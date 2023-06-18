from copy import copy
from enum import Enum
import typing as tp
from functools import partial

import os
import polars as pl

from multiprocessing import Pool

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from utils import (
    BASE_DATA_PATH, MAX_OCR_LEN, BLANK_SYMBOL,
    get_tokens_map, get_language_stats, to_tokens, to_caption,
    load_image,
)


class DatasetSplit(Enum):
    Train = 0
    Val = 1
    Test = 2

    def __str__(self):
        return self.name.lower()
    

class OCRDataset(Dataset):
    def __init__(
        self, root: str, split: DatasetSplit, meta: pl.DataFrame,
        chars_to_idx: tp.Dict[str, int], idx_to_chars: tp.Dict[int, str],
        transform: tp.Callable[[torch.Tensor], torch.Tensor],
        sample_proba: float = None, seed: int = 0,
        preload_data: bool = False, preload_workers: int = 8,
    ):
        assert split == DatasetSplit.Train or split == DatasetSplit.Test
        self._images_path = os.path.join(root, str(split), str(split))
        self._split = split
        self._chars_to_idx = chars_to_idx
        self._idx_to_chars = idx_to_chars

        self._split_meta = meta
        if split == DatasetSplit.Test:
            self._split_meta = self._split_meta.with_columns(pl.lit('').alias('Expected'))
        self._split_meta = self._split_meta.with_columns(
            tokens=pl.col('Expected').map(
                lambda series: pl.Series(values=to_tokens(
                    series.to_numpy(), char_to_idx=chars_to_idx#, max_len=MAX_OCR_LEN,
                ))
            ),
            length=self._split_meta.get_column('Expected').apply(lambda caption: len(caption)),
        )

        self._seed = seed
        if sample_proba is not None:
            self._split_meta = self._split_meta.sample(
                fraction=sample_proba, shuffle=True, seed=self._seed)
        self._transform = transform
        self._preload_data = preload_data
        self._load_img_op = partial(load_image, base_path=self._images_path)
        if self._preload_data:
            print('Preloading data')
            with Pool(preload_workers) as pool:
                self._images = pool.map(self._load_img_op, self._split_meta.get_column('Id').to_list())

    def print_data_stats(self):
        print(f'Info for split {str(self._split)}:')
        print(f'Samples: {len(self._split_meta)}')
        if 'tokens' in self._split_meta:
            print('Sequence lengths:',
                self._split_meta.get_column('Expected').apply(lambda tokens: len(tokens)).describe())

    def get_true_train_val_splits(self, val_size: float):
        assert self._split == DatasetSplit.Train
        assert 0 <= val_size and val_size <= 1
        pivot = int(len(self) * (1 - val_size))
        shuffled_meta = self._split_meta.sample(fraction=1, shuffle=True, seed=self._seed)
        train_split = copy(self)
        val_split = copy(self)
        train_split._split_meta = shuffled_meta.head(pivot)
        train_split._split = DatasetSplit.Train
        val_split._split_meta = shuffled_meta.tail(-pivot)
        val_split._split = DatasetSplit.Val
        return train_split, val_split

    def __getitem__(self, idx):
        meta_row = self._split_meta.row(idx, named=True)
        image = (
            self._load_img_op(meta_row['Id'])
            if not self._preload_data else self._images[idx]
        )
        return (
            self._transform(image), torch.atleast_1d(torch.tensor(meta_row['tokens'])),
            meta_row['length'], meta_row['Expected'],
        )

    def __len__(self):
        return self._split_meta.shape[0]

    @property
    def blank_symbol(self):
        return self._chars_to_idx[BLANK_SYMBOL]

    def encode(self, s):
        return to_tokens(s, self._chars_to_idx, MAX_OCR_LEN)

    def decode(self, tokens):
        return to_caption(tokens, self._idx_to_chars, blank_idx=self.blank_symbol)


if __name__ == '__main__':
    meta = pl.read_csv(os.path.join(BASE_DATA_PATH, 'train_labels.csv'))
    meta = meta.drop_nulls().filter(pl.col('Expected').apply(len).le(MAX_OCR_LEN))
    print(get_language_stats(meta))
    chars_to_idx, idx_to_chars = get_tokens_map(meta)
    data = OCRDataset(
        BASE_DATA_PATH, DatasetSplit.Train, meta,
        chars_to_idx, idx_to_chars, T.ToTensor(), None
    )
    train, val = data.get_true_train_val_splits(0.2)
    print(train[0])
    print(val[0])
    assert data.decode(data.encode(['Абырвалг'])) == ['Абырвалг']
