from argparse import ArgumentParser, Namespace
import os
import numpy as np

from torch.utils.data import DataLoader

from dataset import SportsClassificationImageDataset, DatasetSplit, construct_label_to_class_idx
from utils import BASE_DATA_PATH, predict, load_model


def main(model_path: str):
    label_to_class_idx = construct_label_to_class_idx(os.path.join(BASE_DATA_PATH, 'train.csv'))
    model, train_config = load_model(model_path, {'n_classes': len(label_to_class_idx)})
    model.eval()

    is_gpu = train_config.device == 'cuda'

    if is_gpu and train_config.train_data_parallel:
        raise NotImplementedError()
    else:
        model = model.to(train_config.device)

    model_input_transform = model.get_input_transform()
    test_data = SportsClassificationImageDataset(
        BASE_DATA_PATH, DatasetSplit.Test, label_to_class_idx, model_input_transform,
    )
    test_loader = DataLoader(
        test_data, train_config.batch_size,
        shuffle=False, num_workers=4,
        pin_memory=is_gpu,
    )
    probas, gt, entropy = predict(model, test_loader, train_config)
    test_predictions = test_data.combine_with_predictions(np.argmax(probas, axis=-1))
    test_predictions.to_csv('test_predictions.csv', index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--saved-model-path', type=str, required=True)
    args = parser.parse_args()
    path = (
        args.saved_model_path
        if args.saved_model_path.startswith('/')
        else os.path.join(os.getcwd(), args.saved_model_path)
    )
    main(path)
    