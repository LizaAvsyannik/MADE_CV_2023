import os
import typing as tp

from tqdm import tqdm

from dataclasses import dataclass, asdict

import torch
from torch import nn
from torch.utils.data import DataLoader

from models import MODEL_NAME_TO_CLASS


BASE_DATA_PATH = os.path.join(os.getcwd(), 'data')


@dataclass
class TrainConfig:
    backbone: str
    device: str
    n_epochs: int
    batch_size: int
    val_size: float
    start_lr: float
    backbone_lr: float
    clip_grad_norm: float
    dataset_sample_proba: float
    metrics: tp.List[str]
    artifacts_path: str
    train_data_parallel: bool = False

    def __str__(self):
        params = asdict(self)
        params.pop('artifacts_path')
        return '-'.join(map(str, params.values()))


def load_model(path: str, model_ctor_args: tp.Dict[str, tp.Any]):
    state_dict = torch.load(path)
    train_config = TrainConfig(**state_dict['config'])
    model = MODEL_NAME_TO_CLASS[train_config.backbone](**model_ctor_args)
    model.load_state_dict(state_dict['model'])
    return model, train_config


@torch.no_grad()
def predict(model: nn.Module, data: DataLoader, train_config: TrainConfig):
    predictions_history = []
    gt_history = []
    entropy_history = []
    for images, labels, raw_labels in tqdm(data, desc='predict'):
        predictions = model(images.to(train_config.device))
        probas = torch.exp(predictions)
        entropy_history.extend(
            torch.sum(-probas * torch.log(probas), dim=-1).cpu().tolist()
        )
        predictions_history.extend(probas.cpu().tolist())
        gt_history.extend(labels.tolist())
    
    return predictions_history, gt_history, entropy_history
