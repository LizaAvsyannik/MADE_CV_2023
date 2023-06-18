import typing as tp
from argparse import ArgumentParser, Namespace

import datetime
import os

import multiprocessing

from tqdm import tqdm

import numpy as np
import polars as pl

from matplotlib import pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from torchvision import transforms as T

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from dataset import OCRDataset, DatasetSplit
from models import MODEL_NAME_TO_CLASS
from metrics import METRIC_NAMES_TO_CLASSES
from utils import (
    BASE_DATA_PATH, TrainConfig, MAX_OCR_LEN,
    get_tokens_map, get_language_stats, predict,
    is_distributed_training, is_main_node,
    get_augmentation_transforms,
)

DEBUG = True


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


def save_checkpoint(
    model: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler,
    train_config: TrainConfig, experiment_name: str, epoch: int, suffix: str,
):
    if not is_main_node():
        return

    if is_distributed_training():
        model = model.module

    n_classes = model._n_classes

    artifacts_path = os.path.expanduser(os.path.join(train_config.artifacts_path, experiment_name))
    os.makedirs(artifacts_path, exist_ok=True)
    torch.save(
        {
            'model': model.state_dict(),
            'n_classes': n_classes,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'config': train_config.asdict(),
            'epoch': epoch,
        },
        os.path.join(artifacts_path, f'checkpoint_{suffix}.pth'),
    )


def load_checkpoint(
    model: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler,
    checkpoint_path: str, train_config: TrainConfig,
):
    state_dict = torch.load(checkpoint_path, map_location=train_config.device)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    scheduler.load_state_dict(state_dict['scheduler'])

    loaded_config = state_dict['config']
    loaded_config['n_epochs'] = train_config.n_epochs
    loaded_config['device'] = train_config.device
    assert loaded_config == train_config.asdict(), f'{loaded_config}\n{train_config.asdict()}'

    return state_dict['epoch']


def train_single_epoch(
    model: nn.Module, train_data: DataLoader,
    optimizer: optim.Optimizer, train_config: TrainConfig,
    writer: SummaryWriter, epoch: int,
):
    entropy_history = []
    loss_history = []
    predictions_history = []
    gt_history = []
    edit_distance = METRIC_NAMES_TO_CLASSES['mean_edit_distance']()
    loss_fn = nn.CTCLoss()
    for batch_idx, (images, tokens, target_lengths, captions) in tqdm(
        enumerate(train_data), desc='train', position=1, disable=not is_main_node()
    ):
        optimizer.zero_grad()
        log_probas = model(images.to(train_config.device))
        probas = torch.exp(log_probas)
        predictions = probas.argmax(-1).T

        predicted_captions = train_data.dataset.decode(predictions.detach().cpu().tolist())
        predictions_history.extend(predicted_captions)
        gt_history.extend(captions)
        entropy_history.extend(
            torch.mean(torch.sum(-probas * log_probas, dim=-1), dim=0).cpu().tolist()
        )

        if DEBUG and batch_idx == 0 and is_main_node():
            os.makedirs(f'images/{epoch}', exist_ok=True)
            plt.imshow(images[0].permute(1, 2, 0).numpy())
            plt.title(f'Predicted: {predicted_captions[0]}; GT: {captions[0]}')
            plt.savefig(f'images/{epoch}/{batch_idx}.png')

        input_lengths = torch.full(
            (images.shape[0],), log_probas.shape[0], dtype=torch.int32)
        loss = loss_fn(
            log_probas, tokens.int(), input_lengths.int(), target_lengths.int(),
        )
        loss_history.append(loss.cpu().detach().item())
        loss.backward()
        for param_group in optimizer.param_groups:
            nn.utils.clip_grad.clip_grad_norm_(param_group['params'], train_config.clip_grad_norm)
        optimizer.step()

    if is_main_node():
        print(list(zip(predictions_history[-5:], gt_history[-5:])))
        print(np.mean(loss_history))
        writer.add_scalar('train/loss', np.mean(loss_history), epoch)
        writer.add_scalar('train/edit_distance', edit_distance(predictions_history, gt_history), epoch)
        writer.add_scalar('train/entropy', np.mean(entropy_history), epoch)


def train(
    model: nn.Module, train_data: DataLoader, val_data: DataLoader,
    optimizer: optim.Optimizer, lr_scheduler: optim.lr_scheduler._LRScheduler,
    train_config: TrainConfig, writer: SummaryWriter, experiment_name: str,
    start_epoch: int = 0,
):
    for epoch in tqdm(range(start_epoch, train_config.n_epochs), desc='epoch', position=0):
        if is_distributed_training():
            train_data.sampler.set_epoch(epoch)
            dist.barrier()

        model.train()
        train_single_epoch(model, train_data, optimizer, train_config, writer, epoch)

        if is_main_node():
            model.eval()
            metrics = evaluate_model(
                # See https://github.com/pytorch/pytorch/issues/54059
                model.module if is_distributed_training() else model,
                val_data, train_config)
            print(metrics)
            for k, v in metrics.items():
                writer.add_scalar(f'eval/{k}', v, epoch)
                lr_scheduler.step(next(iter(metrics.values())))

        if (epoch + 1) % train_config.checkpoint_frequency == 0:
            save_checkpoint(
                model, optimizer, lr_scheduler,
                train_config, experiment_name, epoch, f'{epoch}')
    return model


def get_train_loaders(
    train_config: TrainConfig, meta: pl.DataFrame,
    chars_to_idx: tp.Dict[str, int], idx_to_chars: tp.Dict[int, str],
    transform,
):
    is_gpu = train_config.device != 'cpu'
    image_transform = get_augmentation_transforms(transform)
    train_data, val_data = OCRDataset(
        BASE_DATA_PATH, DatasetSplit.Train, meta, chars_to_idx, idx_to_chars,
        image_transform, sample_proba=train_config.dataset_sample_proba,
        preload_data=train_config.preload_data, preload_workers=train_config.preload_workers,
    ).get_true_train_val_splits(train_config.val_size)
    if is_main_node():
        train_data.print_data_stats()
        val_data.print_data_stats()

    sampler = DistributedSampler(train_data) if is_distributed_training() else None
    train_loader = DataLoader(
        train_data, train_config.batch_size,
        shuffle=(sampler is None), num_workers=8,
        pin_memory=is_gpu,
        pin_memory_device=train_config.device,
        sampler=sampler,
    )
    val_loader = DataLoader(
        val_data, train_config.batch_size,
        shuffle=False, num_workers=8,
        pin_memory=is_gpu,
        pin_memory_device=train_config.device,
    )
    
    return train_loader, val_loader


def main(train_config: TrainConfig):
    is_distributed = 'MASTER_ADDR' in os.environ
    if is_distributed:
        if is_main_node():
            print('Training in distributed context')
        multiprocessing.set_start_method('spawn')
        dist.init_process_group("nccl")

    datetime_prefix = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    experiment_name = f'{datetime_prefix}-{str(train_config)}'

    board_writer = SummaryWriter(f'runs/{experiment_name}') if is_main_node() else None
    meta = pl.read_csv(os.path.join(BASE_DATA_PATH, 'train_labels.csv'))
    meta = meta.drop_nulls().filter(pl.col('Expected').apply(len).le(MAX_OCR_LEN))

    if train_config.print_language_stats and is_main_node():
        print(get_language_stats(meta))

    chars_to_idx, idx_to_chars = get_tokens_map(meta)

    model = MODEL_NAME_TO_CLASS[train_config.backbone](len(chars_to_idx))
    model_input_transform = model.get_input_transform()
    model = model.to(train_config.device)

    optimizer = optim.AdamW(model.parameters(), lr=train_config.start_lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.333, patience=4, cooldown=1, min_lr=1e-6,
    )
    start_epoch = 0
    if train_config.checkpoint_path:
        start_epoch = load_checkpoint(model, optimizer, lr_scheduler, train_config.checkpoint_path, train_config)
    if is_distributed:
        model = DDP(model, device_ids=[train_config.device], output_device=train_config.device)

    # TODO: augmentations
    train_loader, val_loader = get_train_loaders(
        train_config, meta, chars_to_idx, idx_to_chars, model_input_transform,
    )
    trained_model = train(
        model, train_loader, val_loader,
        optimizer, lr_scheduler,
        train_config, board_writer,
        experiment_name, start_epoch,
    )
    save_checkpoint(
        trained_model, optimizer, lr_scheduler,
        train_config, experiment_name, train_config.n_epochs, 'finish')
    if is_distributed:
        dist.destroy_process_group()



def add_arguments(parser: ArgumentParser):
    parser.add_argument(
        '--backbone', type=str, choices=list(MODEL_NAME_TO_CLASS.keys()), required=True,
    )

    parser.add_argument('--device', type=str, default=f'cuda:{os.environ.get("LOCAL_RANK")}' or 'cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n-epochs', type=int, default=100)
    parser.add_argument('--start-lr', type=float, default=1e-3)
    parser.add_argument('--clip-grad-norm', type=float, default=5.0)
    parser.add_argument(
        '--val-size', type=float, default=0.1, help='Validation dataset fraction of train',
    )
    parser.add_argument(
        '--dataset-sample-proba', type=float, default=None,
        help='If provided, subsample datasets to specified factor',
    )
    parser.add_argument('--metrics', type=str, nargs='*', default=[])
    parser.add_argument(
        '--artifacts-path', type=str, default=os.path.join(os.getcwd(), 'artifacts'),
    )
    parser.add_argument('--print-language-stats', action='store_true')

    parser.add_argument('--preload-data', action='store_true')
    parser.add_argument('--preload-workers', type=int, default=8)

    parser.add_argument('--checkpoint-path', type=str, default=None)
    parser.add_argument('--checkpoint-frequency', type=int, default=20)


def config_from_args(args: Namespace):
    return TrainConfig(**vars(args))


if __name__ == '__main__':
    parser = ArgumentParser()
    add_arguments(parser)
    main(config_from_args(parser.parse_args()))
