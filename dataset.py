from copy import copy
from enum import Enum
import typing as tp

import os
import numpy as np
import pandas as pd

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from utils import BASE_DATA_PATH


def construct_label_to_class_idx(meta_path):
    meta = pd.read_csv(meta_path)
    labels = meta['label'].unique().tolist()
    return dict(zip(labels, range(len(labels))))


class DatasetSplit(Enum):
    Train = 0
    Val = 1
    Test = 2

    def __str__(self):
        return self.name.lower()


class SportsClassificationImageDataset(Dataset):
    def __init__(
        self, root: str, split: DatasetSplit, label_to_class_idx: dict,
        transform: tp.Callable[[torch.Tensor], torch.Tensor],
        sample_proba: float = None,
    ):
        assert split == DatasetSplit.Train or split == DatasetSplit.Test
        self._images_path = os.path.join(root, str(split))
        self._split = split
        self._split_meta = pd.read_csv(os.path.join(root, f'{split}.csv'))
        if sample_proba is not None:
            self._split_meta = self._split_meta.sample(frac=sample_proba)
        self._label_to_class_idx = label_to_class_idx
        self._class_idx_to_label = {v: k for k, v in self._label_to_class_idx.items()}
        self._transform = transform
        if split == DatasetSplit.Train:
            self._split_meta['class_idx'] = self._split_meta.apply(
                lambda row: self._label_to_class_idx[row['label']],
                axis=1,
            )
        else:
            self._split_meta['label'] = ''
            self._split_meta['class_idx'] = -1

    def map_classes_to_labels(self, classes: tp.List[int]) -> tp.List[str]:
        return [self._class_idx_to_label[idx] for idx in classes]

    def combine_with_predictions(self, predictions: tp.List[int]) -> pd.DataFrame:
        df = copy(self._split_meta)
        df['label'] = self.map_classes_to_labels(predictions)
        return df[['image_id', 'label']]

    def print_data_stats(self):
        print(f'Info for split {str(self._split)}:')
        print(f'Samples: {len(self._split_meta)}')
        if 'label' in self._split_meta:
            print(self._split_meta.groupby('label').count())

    def get_true_train_val_splits(self, val_size: float):
        assert self._split == DatasetSplit.Train
        assert 0 <= val_size and val_size <= 1
        shuffled = np.random.permutation(len(self))
        pivot = int(len(self) * (1 - val_size))
        train_split = copy(self)
        val_split = copy(self)
        train_split._split_meta = self._split_meta.iloc[shuffled[:pivot]]
        train_split._split = DatasetSplit.Train
        val_split._split_meta = self._split_meta.iloc[shuffled[pivot:]]
        val_split._split = DatasetSplit.Val
        return train_split, val_split

    def __getitem__(self, idx):
        meta_row = self._split_meta.iloc[idx]
        img = Image.open(os.path.join(self._images_path, meta_row['image_id']))
        img = img.convert('RGB')
        return self._transform(img), meta_row['class_idx'], meta_row['label']


    def __len__(self):
        return len(self._split_meta)
    

if __name__ == '__main__':
    data = SportsClassificationImageDataset(
        BASE_DATA_PATH, DatasetSplit.Train,
        construct_label_to_class_idx(os.path.join(BASE_DATA_PATH, 'train.csv')),
        T.Compose([T.ToTensor()])
    )
    train, val = data.get_true_train_val_splits(0.3)
    print(len(train), len(val))

    first_element = data[0]
    print(first_element[0].shape)
