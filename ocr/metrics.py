import numpy as np

from torchaudio.functional import edit_distance


METRIC_NAMES_TO_CLASSES = {
    'mean_edit_distance': lambda: lambda predictions, gt: 
        np.mean([edit_distance(pred_caption, gt_caption)
                 for pred_caption, gt_caption in zip(predictions, gt)]),
}
