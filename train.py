from argparse import ArgumentParser, Namespace

import datetime
import os

from dataclasses import asdict
from tqdm import tqdm

import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from dataset import SportsClassificationImageDataset, DatasetSplit, construct_label_to_class_idx
from models import MODEL_NAME_TO_CLASS
from metrics import METRIC_NAMES_TO_CLASSES
from utils import BASE_DATA_PATH, TrainConfig, predict


def evaluate_model(model: nn.Module, val_data: DataLoader, train_config: TrainConfig):
    metrics_to_calculate = {
        metric_name: METRIC_NAMES_TO_CLASSES[metric_name]()
        for metric_name in train_config.metrics
    }
    predictions_history, gt_history, entropy = predict(model, val_data, train_config)
    metrics = {'entropy': np.mean(entropy)}
    metrics.update({
        name: metric(predictions_history, gt_history)
        for name, metric in metrics_to_calculate.items()
    })
    return metrics


def train_single_epoch(
    model: nn.Module, train_data: DataLoader, val_data: DataLoader,
    optimizer: optim.Optimizer, train_config: TrainConfig,
    writer: SummaryWriter, epoch: int,
):
    entropy_history = []
    loss_history = []
    loss_fn = nn.NLLLoss()
    for images, labels, raw_labels in tqdm(train_data, desc='train', position=1):
        optimizer.zero_grad()
        predictions = model(images.to(train_config.device))
        probas = torch.exp(predictions)
        entropy_history.extend(
            torch.sum(-probas * torch.log(probas), dim=-1).cpu().tolist()
        )
        loss = loss_fn(predictions, labels.to(train_config.device))
        loss_history.append(loss.cpu().detach().item())
        loss.backward()
        for param_group in optimizer.param_groups:
            nn.utils.clip_grad.clip_grad_norm_(param_group['params'], train_config.clip_grad_norm)
        optimizer.step()
    writer.add_scalar('train/loss', np.mean(loss_history), epoch)
    writer.add_scalar('train/entropy', np.mean(entropy_history), epoch)


def train(
    model: nn.Module, train_data: DataLoader, val_data: DataLoader,
    optimizer: optim.Optimizer, lr_scheduler: optim.lr_scheduler.LRScheduler,
    train_config: TrainConfig, writer: SummaryWriter,
):
    model.train()
    for epoch in tqdm(range(train_config.n_epochs), desc='epoch', position=0):
        train_single_epoch(model, train_data, val_data, optimizer, train_config, writer, epoch)
        metrics = evaluate_model(model, val_data, train_config)
        print(metrics)
        for k, v in metrics.items():
            writer.add_scalar(f'eval/{k}', v, epoch)
        if 'nll_loss' in metrics:
            lr_scheduler.step(metrics['nll_loss'])
        else:
            lr_scheduler.step(next(iter(metrics.values())))
    return model


def get_train_loaders(train_config: TrainConfig, class_to_idx: dict, transform):
    is_gpu = train_config.device == 'cuda'
    train_data, val_data = SportsClassificationImageDataset(
        BASE_DATA_PATH, DatasetSplit.Train, class_to_idx, transform,
        sample_proba=train_config.dataset_sample_proba,
    ).get_true_train_val_splits(train_config.val_size)
    train_data.print_data_stats()
    val_data.print_data_stats()

    train_loader = DataLoader(
        train_data, train_config.batch_size,
        shuffle=True, num_workers=4,
        pin_memory=is_gpu,
    )
    val_loader = DataLoader(
        val_data, train_config.batch_size,
        shuffle=False, num_workers=4,
        pin_memory=is_gpu,
    )
    return train_loader, val_loader


def main(train_config: TrainConfig):
    is_gpu = train_config.device == 'cuda'

    datetime_prefix = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    experiment_name = f'{datetime_prefix}-{str(train_config)}'

    board_writer = SummaryWriter(f'runs/{experiment_name}')
    label_to_class_idx = construct_label_to_class_idx(os.path.join(BASE_DATA_PATH, 'train.csv'))
    model = MODEL_NAME_TO_CLASS[train_config.backbone](len(label_to_class_idx))
    if is_gpu and train_config.train_data_parallel:
        raise NotImplementedError()
    else:
        model = model.to(train_config.device)

    model_input_transform = model.get_input_transform()
    train_loader, val_loader = get_train_loaders(
        train_config, label_to_class_idx, model_input_transform,
    )

    optimizer_param_groups = [
        {'params': model.get_transfer_trainable_parameters(), 'lr': train_config.start_lr},
        {'params': model.get_backbone_trainable_parameters(), 'lr': train_config.backbone_lr},
    ]
    optimizer = optim.Adam(optimizer_param_groups)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.333, patience=4, cooldown=1, min_lr=[1e-6, 1e-8],
    )
    trained_model = train(
        model, train_loader, val_loader,
        optimizer, lr_scheduler,
        train_config, board_writer,
    )

    artifacts_path = os.path.expanduser(os.path.join(train_config.artifacts_path, experiment_name))
    os.makedirs(artifacts_path, exist_ok=True)
    torch.save(
        {
            'model': trained_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
            'config': asdict(train_config),
        },
        os.path.join(artifacts_path, 'trained.pth'),
    )


def add_arguments(parser: ArgumentParser):
    parser.add_argument(
        '--backbone', type=str, choices=list(MODEL_NAME_TO_CLASS.keys()), required=True,
    )
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--train-data-parallel', action='store_true')  # not implemented
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n-epochs', type=int, default=100)
    parser.add_argument('--start-lr', type=float, default=1e-3)
    parser.add_argument('--backbone-lr', type=float, default=1e-5)
    parser.add_argument('--clip-grad-norm', type=float, default=5.0)
    parser.add_argument(
        '--val-size', type=float, default=0.1, help='Validation dataset fraction of train',
    )
    parser.add_argument(
        '--dataset-sample-proba', type=float, default=None,
        help='If provided, subsample datasets to specified factor',
    )
    parser.add_argument('--metrics', type=str, nargs='+', default=['micro_f1', 'nll_loss'])
    parser.add_argument(
        '--artifacts-path', type=str, default=os.path.join(os.getcwd(), 'artifacts'),
    )


def config_from_args(args: Namespace):
    return TrainConfig(**vars(args))


if __name__ == '__main__':
    parser = ArgumentParser()
    add_arguments(parser)
    main(config_from_args(parser.parse_args([
        '--backbone', 'swin_v2_s', '--device', 'cuda', '--batch-size', '512',
        '--n-epochs', '50',
        '--val-size', '0.005',
        # '--dataset-sample-proba', '0.05',
    ])))
