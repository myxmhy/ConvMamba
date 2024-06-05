import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def rescale(x):
    return (x - x.max()) / (x.max() - x.min()) * 2 - 1


def MAE(pred, label):
    return np.mean(np.abs(pred-label))


def MRE(pred, label):
    return np.mean(np.abs(pred-label) / (np.abs(label) + 1e-20))


def MSE(pred, label):
    return np.mean((pred-label)**2)


def RMSE(pred, label):
    return np.sqrt(np.mean((pred-label)**2))


def MXRE(pred, label):
    return np.max(np.abs(pred-label) / (np.abs(label) + 1e-20))


def ACC(pred, label):
    return accuracy_score(label, pred)


def PREC(pred, label):
    return precision_score(label, pred, average='macro', zero_division=0)


def RECALL(pred, label):
    return recall_score(label, pred, average='macro', zero_division=0)


def F1(pred, label):
    return f1_score(label, pred, average='macro', zero_division=0)


def metric(pred, label, metrics=['acc', 'prec', 'recall', 'f1'],
            return_log=True):
    """The evaluation function to output metrics.

    Args:
        pred (tensor): The prediction values of output prediction.
        label (tensor): The prediction values of output prediction.
        metric (str | list[str]): Metrics to be evaluated.
        return_log (bool): Whether to return the log string.

    Returns:
        dict: evaluation results
    """
    eval_log = ""
    allowed_metrics = ['mae', 'mse', 'rmse', 'ssim', 'mre', 'mxre',
                       'acc', 'prec', 'recall', 'f1']
    invalid_metrics = set(metrics) - set(allowed_metrics)
    if len(invalid_metrics) != 0:
        raise ValueError(f'metric {invalid_metrics} is not supported.')
    
    eval_res = {}

    if 'mse' in metrics:
        eval_res['mse'] = MSE(pred, label)

    if 'mae' in metrics:
        eval_res['mae'] = MAE(pred, label)

    if 'rmse' in metrics:
        eval_res['rmse'] = RMSE(pred, label)

    if 'mre' in metrics:
        eval_res['mre'] = MRE(pred, label)
    
    if 'mxre' in metrics:
        eval_res['mxre'] = MXRE(pred, label)

    if 'acc' in metrics:
        eval_res['acc'] = ACC(pred, label)

    if 'prec' in metrics:
        eval_res['prec'] = PREC(pred, label)

    if 'recall' in metrics:
        eval_res['recall'] = RECALL(pred, label)
    
    if 'f1' in metrics:
        eval_res['f1'] = F1(pred, label)

    if return_log:
        for k, v in eval_res.items():
            eval_str = f"{k}:{v:.5e}" if len(eval_log) == 0 else f", {k}:{v:.5e}"
            eval_log += eval_str
    return eval_res, eval_log
