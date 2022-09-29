import numpy as np
from functools import partial
from collections import Counter
from joblib import Parallel, delayed
from sklearn import metrics
from scipy.interpolate import interp1d


def voxel_scale_metrics(y_pred: np.ndarray, y: np.ndarray, r: float = 0.001):
    """
    Compute DSC, nDSC, TPR, TDR, FNR.
    :param y_pred:
    :param y:
    :param r: effective lesion load
    :return: dict
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


def voxel_scale_rc(y_pred: np.ndarray, y: np.ndarray, uncertainties: np.ndarray, fracs_retained: np.ndarray, metric_name: str,
                   n_jobs: int = None):
    """
    Compute error retention curve values and nDSC-AAC on voxel-scale.
    :param fracs_retained:
    :param y:
    :param y_pred:
    :param uncertainties:
    :param n_jobs: number of parallel processes
    :return: tuple with nDSC-AAC and nDSC values of the error retention curve
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


def lesion_scale_rc(tp_les_uncs: list, fp_les_uncs: list, fn_les_uncs: list, fracs_retained: np.ndarray):
    """
    Compute error retention curve values and F1les-AAC on lesion-scale including fasle negative lesions.
    :param tp_les_uncs:
    :param fp_les_uncs:
    :param fn_les_uncs:
    :type fracs_retained: numpy.ndarray [n_fr,]
    :return: F1les-AAC, values of error retention curve, values of error retention curve before interpolation.
    :rtype: tuple
    """

    def compute_f1(lesion_types):
        counter = Counter(lesion_types)
        if counter['tp'] + 0.5 * (counter['fp'] + counter['fn']) == 0.0:
            return 0
        return counter['tp'] / (counter['tp'] + 0.5 * (counter['fp'] + counter['fn']))

    # list of lesions uncertainties tp, fp, fn
    uncs_list = [np.asarray(tp_les_uncs),
                 np.asarray(fp_les_uncs),
                 np.asarray(fn_les_uncs)]
    # list of lesion types
    lesion_type_list = [np.full(shape=unc.shape, fill_value=fill)
                        for unc, fill in zip(uncs_list, ['tp', 'fp', 'fn'])
                        if unc.size > 0]
    uncs_list = [unc for unc in uncs_list if unc.size > 0]
    uncs_all = np.concatenate(uncs_list, axis=0)
    lesion_type_all = np.concatenate(lesion_type_list, axis=0)

    del uncs_list, lesion_type_list
    assert uncs_all.shape == lesion_type_all.shape

    # sort uncertainties and lesions types
    ordering = uncs_all.argsort()
    lesion_type_all = lesion_type_all[ordering][::-1]

    f1_values = [compute_f1(lesion_type_all)]

    # reject the most certain lesion
    for i_l, lesion_type in enumerate(lesion_type_all):
        if lesion_type == 'fp':
            lesion_type_all[i_l] = 'tn'
        elif lesion_type == 'fn':
            lesion_type_all[i_l] = 'tp'
        f1_values.append(compute_f1(lesion_type_all))

    # interpolate the curve and make predictions in the retention fraction nodes
    n_lesions = lesion_type_all.shape[0]
    spline_interpolator = interp1d(x=[_ / n_lesions for _ in range(n_lesions + 1)],
                                   y=f1_values[::-1],
                                   kind='slinear', fill_value="extrapolate")
    f1_interp = spline_interpolator(fracs_retained)
    return 1. - metrics.auc(fracs_retained, np.asarray(f1_interp)), f1_interp


def lesion_scale_rc_nofn(tp_les_uncs: list, fp_les_uncs: list, n_fn: int, fracs_retained: np.ndarray):
    """
    Compute error retention curve values and F1les-AAC on lesion-scale without false negative lesions.
    :type fracs_retained: numpy.ndarray [n_fr,]
    :return: F1les-AAC, values of error retention curve, values of error retention curve before interpolation.
    :rtype: tuple
    """

    def compute_f1(lesion_types, num_fn):
        counter = Counter(lesion_types)
        if counter['tp'] + 0.5 * (counter['fp'] + num_fn) == 0.0:
            return 0.
        return counter['tp'] / (counter['tp'] + 0.5 * (counter['fp'] + num_fn))

    # list of lesions uncertainties tp, fp, fn
    uncs_list = [np.asarray(tp_les_uncs), np.asarray(fp_les_uncs)]
    # list of lesion types
    lesion_type_list = [np.full(shape=unc.shape, fill_value=fill)
                        for unc, fill in zip(uncs_list, ['tp', 'fp'])
                        if unc.size > 0]
    uncs_list = [unc for unc in uncs_list if unc.size > 0]
    uncs_all = np.concatenate(uncs_list, axis=0)
    lesion_type_all = np.concatenate(lesion_type_list, axis=0)

    del uncs_list, lesion_type_list
    assert uncs_all.shape == lesion_type_all.shape

    # sort uncertainties and lesions types
    ordering = uncs_all.argsort()
    lesion_type_all = lesion_type_all[ordering][::-1]

    f1_values = [compute_f1(lesion_type_all, num_fn=n_fn)]

    # reject the most certain lesion
    for i_l, lesion_type in enumerate(lesion_type_all):
        if lesion_type == 'fp':
            lesion_type_all[i_l] = 'tn'
        elif lesion_type == 'fn':
            lesion_type_all[i_l] = 'tp'
        f1_values.append(compute_f1(lesion_type_all, num_fn=n_fn))

    # interpolate the curve and make predictions in the retention fraction nodes
    n_lesions = lesion_type_all.shape[0]
    spline_interpolator = interp1d(x=[_ / n_lesions for _ in range(n_lesions + 1)],
                                   y=f1_values[::-1],
                                   kind='slinear', fill_value="extrapolate")
    f1_interp = spline_interpolator(fracs_retained)
    return 1. - metrics.auc(fracs_retained, np.asarray(f1_interp)), f1_interp
