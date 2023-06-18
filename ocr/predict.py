from argparse import ArgumentParser, Namespace
import os
import numpy as np

import polars as pl

from torch.utils.data import DataLoader

from dataset import OCRDataset, DatasetSplit
from models import load_model
from utils import BASE_DATA_PATH, predict, MAX_OCR_LEN, get_tokens_map


def main(model_path: str, device: str):
    is_gpu = device != 'cpu'
    model, train_config = load_model(model_path, device)
    model.eval()

    train_meta = pl.read_csv(os.path.join(BASE_DATA_PATH, 'train_labels.csv'))
    train_meta = train_meta.drop_nulls().filter(pl.col('Expected').apply(len).le(MAX_OCR_LEN))

    chars_to_idx, idx_to_chars = get_tokens_map(train_meta)

    test_img_names = os.listdir(os.path.join(BASE_DATA_PATH, 'test', 'test'))
    test_meta = pl.DataFrame().with_columns(pl.Series(values=test_img_names).alias('Id'))

    model_input_transform = model.get_input_transform()
    test_data = OCRDataset(
        BASE_DATA_PATH, DatasetSplit.Test, test_meta, chars_to_idx, idx_to_chars, model_input_transform,
    )
    test_loader = DataLoader(
        test_data, train_config.batch_size,
        shuffle=False, num_workers=4,
        pin_memory=is_gpu,
        pin_memory_device=device,
    )
    captions, gt, entropy = predict(model, test_loader, train_config)
    test_predictions = test_meta.with_columns(pl.Series(values=captions).alias('Predicted'))
    test_predictions.write_csv('test_predictions.csv')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--saved-model-path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    path = (
        args.saved_model_path
        if args.saved_model_path.startswith('/')
        else os.path.join(os.getcwd(), args.saved_model_path)
    )
    main(path, args.device)
    