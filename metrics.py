import numpy as np

from sklearn.metrics import f1_score, log_loss


METRIC_NAMES_TO_CLASSES = {
    'micro_f1': lambda: lambda probas, gt: f1_score(
        gt, np.argmax(probas, axis=-1), average='micro',
    ),
    'nll_loss': lambda: lambda probas, gt: log_loss(gt, probas)
}
