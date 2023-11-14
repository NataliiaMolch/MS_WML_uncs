"""
Implementation of voxel and lesion scale retention curves.
"""
import numpy as np
from functools import partial
from collections import Counter
from joblib import Parallel, delayed
from sklearn import metrics
from scipy.interpolate import interp1d


def voxel_scale_metrics(y_pred: np.ndarray, y: np.ndarray, r: float = 0.001) -> dict:
    """
    Compute DSC, nDSC, TPR, TDR, FNR.
    :param y_pred: predicted binary segmentation mask, shape [H, W, D]
    :param y: ground truth binary segmentation mask, shape [H, W, D]
    :param r: effective lesion load for nDSC computation
    :return: dict of voxel-scale metrics
    """
    if np.sum(y) + np.sum(y_pred) == 0:
        return {'DSC': 1.0, 'nDSC': 1.0, 'TPR': 1.0, 'TDR': 1.0, 'FNR': 0.0}
    else:
        k = (1 - r) * np.sum(y) / (r * (len(y.flatten()) - np.sum(y))) if np.sum(y) != 0.0 else 1.0
        tp = np.sum(y * y_pred)
        fp = np.sum(y_pred[y == 0])
        fn = np.sum(y[y_pred == 0])
        return {
            'DSC': 2.0 * tp / (2.0 * tp + fp + fn),
            'nDSC': 2.0 * tp / (2.0 * tp + fp * k + fn),
            'TPR': tp / (tp + fn),
            'TDR': tp / (tp + fp)
        }


def voxel_scale_rc(y_pred: np.ndarray, y: np.ndarray, uncertainties: np.ndarray,
                   fracs_retained: np.ndarray = np.linspace(0.0, 1.0, 400),
                   metric_name: str = 'DSC', n_jobs: int = None) -> tuple:
    """
    Compute `metric_name` retention curve values and nDSC-AAC on voxel-scale.
    :param y: flattened ground truth binary lesion mask, shape [H*W*D,]
    :param y_pred: flattened predicted binary lesion mask, shape [H*W*D,]
    :param uncertainties: flattened voxel-scale uncertainty map, shape [H*W*D]
    :param fracs_retained: array with fractions of retained voxels, shape [N,]
    :param metric_name: name of metric to be used for building a retention curve, from the ones returned by `voxel_scale_metrics()`
    :param n_jobs: number of parallel processes
    :return: tuple with `metric_name`-AAC (float) and `metric_name`-RC values corresponding to each value of `fracs_retained`, shape [N,]
    """

    def compute_metric(frac_, preds_, gts_, N_, metric_name_):
        pos = int(N_ * frac_)
        curr_preds = preds if pos == N_ else np.concatenate([preds_[:pos], gts_[pos:]])
        return voxel_scale_metrics(y=gts_, y_pred=curr_preds)[metric_name_]

    ordering = uncertainties.argsort()
    gts = y[ordering].copy()
    preds = y_pred[ordering].copy()

    process = partial(compute_metric, preds_=preds, gts_=gts, N_=len(gts), metric_name_=metric_name)
    with Parallel(n_jobs=n_jobs) as parallel_backend:
        dsc_norm_scores = parallel_backend(delayed(process)(frac) for frac in fracs_retained)

    return 1. - metrics.auc(fracs_retained, np.asarray(dsc_norm_scores)), dsc_norm_scores


def lesion_scale_rc(lesion_uncertainties: np.ndarray, lesion_types: np.ndarray,
                    fracs_retained: np.ndarray = np.linspace(0.0, 1.0, 400)) -> tuple:
    """
    Compute lesion F1-retention curve values and lesion F1-AAC on lesion-scale including false negative lesions.
    :param lesion_uncertainties: uncertainty values for each TP, FP and FN lesion of a particular scan, shape [N_TP + N_FP + N_FN,]
    :param lesion_types: array with lesion types, i.e. 'tp', 'fp', 'fn', corresponding to `lesion_uncertainties` array, shape [N_TP + N_FP + N_FN,]
    :param fracs_retained: array with fractions of retained voxels, shape [N,]
    :return: F1les-AAC, values of error retention curve, values of error retention curve before interpolation.
    :return: tuple with lesion F1-AAC (float) and lesion F1-RC values corresponding to each value of `fracs_retained`, shape [N,]
    """

    def compute_f1(lt):
        counter = Counter(lt)
        if counter['tp'] + 0.5 * (counter['fp'] + counter['fn']) == 0.0:
            return 0
        return counter['tp'] / (counter['tp'] + 0.5 * (counter['fp'] + counter['fn']))

    assert lesion_uncertainties.shape == lesion_types.shape

    # sort uncertainties and lesions types
    ordering = lesion_uncertainties.argsort()
    lesion_types = lesion_types[ordering][::-1]

    f1_values = [compute_f1(lesion_types)]
    # reject the most uncertain lesion
    for i_l, lesion_type in enumerate(lesion_types):
        if lesion_type == 'fp':
            lesion_types[i_l] = 'tn'
        elif lesion_type == 'fn':
            lesion_types[i_l] = 'tp'
        f1_values.append(compute_f1(lesion_types))

    # interpolate the curve and make predictions in the retention fraction nodes
    n_lesions = lesion_types.shape[0]
    spline_interpolator = interp1d(x=[_ / n_lesions for _ in range(n_lesions + 1)],
                                   y=f1_values[::-1],
                                   kind='slinear', fill_value="extrapolate")
    f1_interp = spline_interpolator(fracs_retained)
    return 1. - metrics.auc(fracs_retained, np.asarray(f1_interp)), f1_interp


def lesion_scale_rc_nofn(lesion_uncertainties: np.ndarray, lesion_types: np.ndarray, n_fn: int,
                         fracs_retained: np.ndarray = np.linspace(0.0, 1.0, 400)) -> tuple:
    """
    Compute lesion F1-retention curve values and lesion F1-AAC on lesion-scale without false negative lesions.
    :param lesion_uncertainties: uncertainty values for each TP, FP lesion of a particular scan, shape [N_TP + N_FP,]
    :param lesion_types: array with lesion types, i.e. 'tp', 'fp', corresponding to `lesion_uncertainties` array, shape [N_TP + N_FP,]
    :param n_fn: number of false negative lesions in this scan
    :param fracs_retained: array with fractions of retained voxels, shape [N,]
    :return: tuple with lesion F1-AAC (float) and lesion F1-RC values corresponding to each value of `fracs_retained`, shape [N,]
    """

    def compute_f1(lt, num_fn):
        counter = Counter(lt)
        if counter['tp'] + 0.5 * (counter['fp'] + num_fn) == 0.0:
            return 0.
        return counter['tp'] / (counter['tp'] + 0.5 * (counter['fp'] + num_fn))

    assert lesion_uncertainties.shape == lesion_types.shape

    # sort uncertainties and lesions types
    ordering = lesion_uncertainties.argsort()
    lesion_types = lesion_types[ordering][::-1]

    f1_values = [compute_f1(lesion_types, num_fn=n_fn)]

    # reject the most uncertain lesion
    for i_l, lesion_type in enumerate(lesion_types):
        if lesion_type == 'fp':
            lesion_types[i_l] = 'tn'
        f1_values.append(compute_f1(lesion_types, num_fn=n_fn))

    # interpolate the curve and make predictions in the retention fraction nodes
    n_lesions = lesion_types.shape[0]
    spline_interpolator = interp1d(x=[_ / n_lesions for _ in range(n_lesions + 1)],
                                   y=f1_values[::-1],
                                   kind='slinear', fill_value="extrapolate")
    f1_interp = spline_interpolator(fracs_retained)
    return 1. - metrics.auc(fracs_retained, np.asarray(f1_interp)), f1_interp
