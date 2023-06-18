import os
import typing as tp
from functools import partial

from tqdm import tqdm

from dataclasses import dataclass, asdict, field

from lingua import Language, LanguageDetector, LanguageDetectorBuilder

import polars as pl

from PIL import Image

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as T


BASE_DATA_PATH = os.path.join(os.getcwd(), 'data')
MAX_OCR_LEN = 100

BLANK_SYMBOL = 'ע'


def is_distributed_training():
    return 'MASTER_ADDR' in os.environ

def is_main_node():
    return int(os.environ.get('LOCAL_RANK', 0)) == 0


def detect_language(row: dict, detector: LanguageDetector) -> str:
    s: str = row['Expected']
    if s.isdigit():
        return 'digits'
    language = detector.detect_language_of(s)
    if language is None:
        if s.isalnum():
            return 'other_alnum'
        elif s.isascii():
            return 'other_ascii'
        return 'unknown'
    return language.iso_code_639_1.name


def get_tokens_map(meta: pl.DataFrame) -> tp.Tuple[tp.Dict[str, int], tp.Dict[int, str]]:
    chars = set(''.join(meta.get_column('Expected').to_list()))
    assert BLANK_SYMBOL not in chars
    chars = [BLANK_SYMBOL] + list(chars)
    idx_to_chars = dict(enumerate(chars))
    chars_to_idx = {v: k for k, v in idx_to_chars.items()}
    assert chars_to_idx[BLANK_SYMBOL] == 0
    return chars_to_idx, idx_to_chars


def get_language_stats(meta: pl.DataFrame) -> pl.DataFrame:
    languages = [Language.ENGLISH, Language.RUSSIAN]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    meta = meta.with_columns(pl.struct(['Expected']).apply(partial(detect_language, detector=detector)).alias('language'))
    return meta.groupby('language').count()


def load_image(path: str, base_path: str):
    img = Image.open(os.path.join(base_path, path))
    img = img.convert('RGB')
    img.load()
    return img


def get_augmentation_transforms(base_transform):
    return T.Compose([
        T.RandomRotation(degrees=2),
        T.RandomPosterize(bits=4, p=0.2),
        T.RandomApply([T.ColorJitter(.02, .02, .02, .02)], p=0.2),
        base_transform,
        T.RandomApply([T.GaussianBlur(11)], p=0.2),
    ])


@dataclass
class TrainConfig:
    backbone: str
    device: str
    n_epochs: int
    batch_size: int
    val_size: float
    start_lr: float
    clip_grad_norm: float
    dataset_sample_proba: float
    metrics: tp.List[str] = field(default_factory=list)
    artifacts_path: str = ''
    print_language_stats: bool = False

    preload_data: bool = False
    preload_workers: int = 1

    checkpoint_path: str = ''
    checkpoint_frequency: int = 1

    @property
    def transient_fields(self):
        return [
            'metrics', 'artifacts_path', 'print_language_stats',
            'preload_data', 'preload_workers',
            'checkpoint_path', 'checkpoint_frequency',
        ]
    
    def asdict(self):
        params = asdict(self)
        [params.pop(key) for key in self.transient_fields]
        return params


    def __str__(self):
        return '-'.join(map(str, self.asdict().values()))


@torch.no_grad()
def predict(model: nn.Module, data: DataLoader, train_config: TrainConfig):
    predictions_history = []
    gt_history = []
    entropy_history = []
    for images, tokens, _, captions in tqdm(data, desc='predict'):
        log_probas = model(images.to(train_config.device))
        probas = torch.exp(log_probas)
        entropy_history.extend(
            torch.mean(torch.sum(-probas * log_probas, dim=-1), dim=-1).cpu().tolist()
        )
        tokens = probas.argmax(-1).T
        predictions_history.extend(data.dataset.decode(tokens.cpu().tolist()))
        gt_history.extend(captions)
    
    return predictions_history, gt_history, entropy_history


def to_tokens(sequences, char_to_idx, max_len=None, dtype=np.int32):
    max_len = max_len or max(map(len, sequences))
    sequences_ix = np.full(
        (len(sequences), max_len), fill_value=char_to_idx[BLANK_SYMBOL], dtype=dtype,
    )

    for i in range(len(sequences)):
        line_ix = np.array([char_to_idx[c] for c in sequences[i]], dtype=dtype)
        sequences_ix[i, :len(sequences[i])] = line_ix

    return sequences_ix


def to_caption(sequences, idx_to_char, blank_idx=0):  
        def _construct_caption(idx_sequence):
            token_sequence = []
            last_idx = -1
            for idx in idx_sequence:
                token = idx_to_char[idx]
                if idx == blank_idx:
                    last_idx = -1
                    continue
                elif idx == last_idx:
                    continue
                token_sequence.append(token)
                last_idx = idx
            return ''.join(token_sequence)

        return list(map(_construct_caption, sequences))
