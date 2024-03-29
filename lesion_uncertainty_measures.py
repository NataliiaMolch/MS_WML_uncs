"""
Implementation of lesion scale uncertainty meansures.
"""
import numpy as np
from joblib import Parallel, delayed
from functools import partial


vox_uncs_measures = [
    'confidence', 'entropy_of_expected', 'expected_entropy', 'mutual_information', 'epkl', 'reverse_mutual_information'
]
les_uncs_measures = [
    'LSU', 'LSU$^{+}$'
]
for vum in vox_uncs_measures:
    les_uncs_measures.append(f"mean {vum}")
    les_uncs_measures.append(f"logsum {vum}")


def intersection_over_union(mask1, mask2):
    """Compute IoU for 2 binary masks
    mask1 and mask2 should have same dimensions
    """
    return np.sum(mask1 * mask2) / np.sum(mask1 + mask2 - mask1 * mask2)


def single_lesion_uncertainty(cc_mask: np.ndarray, vox_unc_maps: dict, ens_pred_multi: np.ndarray, ens_pred_multi_true: np.ndarray) -> dict:
    """ Compute different uncertainty measures for a single connected component, that include:
    * Mean from vox uncertainty (M Dojat)
    * Log-sum from vox uncertainty (T Alber)
    * Detection disagreement uncertainty (ours)
    :param cc_mask: binary lesion mask, shape [H, W, D]
    :param vox_unc_maps: a dictionary of voxel scale uncertainty maps for different voxel measures of uncertainty, shape [H, W, D]
    :param ens_pred_multi: labeled lesions masks generated by an ensemble of models - threshold tuned for ensemble, shape [M, H, W, D].
    :param ens_pred_multi_true: labeled lesions masks generated by an ensemble of models - thresholds tuned for individual models, shape [M, H, W, D].
    :return: dictionary of lesion uncertainty measures names and values
    :rtype: dict
    """

    def get_max_ious_ccmasks_ens(ens_multi):
        ens_ious_ = []
        for m in range(ens_multi.shape[0]):
            max_iou = 0.0
            for intersec_label in np.unique(cc_mask * ens_multi[m]):
                if intersec_label != 0.0:
                    lesion_m = (ens_multi[m] == intersec_label).astype(float)
                    iou = intersection_over_union(cc_mask, lesion_m)
                    if iou > max_iou:
                        max_iou = iou
            ens_ious_.append(max_iou)
        return ens_ious_

    ''' Check inputs '''
    if not cc_mask.shape == ens_pred_multi[0].shape:
        raise ValueError("Error in input dimensions:\n"
                         f"cc_mask: {cc_mask.shape}, "
                         f"ens_pred_multi: {ens_pred_multi.shape}.")
    res = dict()
    ''' Voxel-scale based uncertainties '''
    for unc_measure, vox_unc_map in vox_unc_maps.items():
        res.update({
            f"mean {unc_measure}": np.sum(vox_unc_map * cc_mask) / np.sum(cc_mask),
            f"logsum {unc_measure}": np.sum(np.log(vox_unc_map[cc_mask == 1]))
        })

    ''' Detection uncertainties '''
    # 1 - mean IoU of connected components on the ensemble's models prediction with cc_mask
    ens_ious = get_max_ious_ccmasks_ens(ens_pred_multi)
    ens_ious_true = get_max_ious_ccmasks_ens(ens_pred_multi_true)

    res.update({
        'LSU': 1 - np.mean(ens_ious), 'LSU$^{+}$': 1 - np.mean(ens_ious_true)
    })
    return res


def lesions_uncertainty(y_pred_multi: np.ndarray, vox_unc_maps: dict, ens_pred_multi: np.ndarray,
                        ens_pred_multi_true: np.ndarray, n_jobs: int = None, dl: bool = True):
    """ Parallel evaluation of all lesion scale uncertainty measures for one subject.
    :param y_pred_multi: labeled lesion mask aka instance segmentation mask, shape [H, W, D]
    :param vox_unc_maps: dict with keys being names of voxel-scale uncertainty measures and values being corresponding 3D uncertainty maps of shape [H, W, D]
    :param ens_pred_multi: labeled lesions masks generated by an ensemble of models - threshold tuned for ensemble, shape [M, H, W, D].
    :param ens_pred_multi_true: labeled lesions masks generated by an ensemble of models - thresholds tuned for individual models, shape [M, H, W, D].
    :param n_jobs: number of parallel workers
    :param dl: if True returns a dictionary of lists, where each key corresponds to a single lesion uncertainty measure,
    list contains uncertainty measures belonging to all lesions.
    :return:
    """

    def ld_to_dl(ld):
        keys = list(ld[0].keys())
        dl = dict(zip(keys, [[] for _ in keys]))
        for el in ld:
            for k in keys:
                v = el[k]
                dl[k].append(v)
        return dl

    cc_labels = np.unique(y_pred_multi)
    cc_labels = cc_labels[cc_labels != 0.0]
    with Parallel(n_jobs=n_jobs) as parallel_backend:
        process = partial(single_lesion_uncertainty,
                          vox_unc_maps=vox_unc_maps,
                          ens_pred_multi_true=ens_pred_multi_true,
                          ens_pred_multi=ens_pred_multi)
        les_uncs_list = parallel_backend(delayed(process)(
            cc_mask=(y_pred_multi == cc_label).astype("float")
        ) for cc_label in cc_labels)  # returns lists of dictionaries, but need dictionary of lists

    if dl:
        if not les_uncs_list:
            metrics = dict(zip(les_uncs_measures, [[] for _ in les_uncs_measures]))
            return metrics
        return ld_to_dl(les_uncs_list)
    else:
        return les_uncs_list


def lesions_uncertainty_maps(y_pred_multi, vox_uncs_map, ens_pred_multi,
                             ens_pred_multi_true: np.ndarray, n_jobs: int = None) -> dict:
    """ Parallel evaluation of all lesion scale uncertainty maps for one subject.
    :param y_pred_multi: labeled lesion mask aka instance segmentation mask
    :param vox_uncs_map: voxel scale lesion uncertainty mask
    :param ens_pred_multi: labeled lesions masks generated by an ensemble of models - threshold tuned for ensemble, shape [M, H, W, D].
    :param ens_pred_multi_true: labeled lesions masks generated by an ensemble of models - thresholds tuned for individual models, shape [M, H, W, D].
    :param n_jobs: number of parallel workers
    :return: dictionary with lesion uncertainty maps for each measure
    :rtype: dict
    """
    cc_labels = np.unique(y_pred_multi)
    cc_labels = cc_labels[cc_labels != 0.0]
    with Parallel(n_jobs=n_jobs) as parallel_backend:
        process = partial(single_lesion_uncertainty,
                          vox_uncs_map=vox_uncs_map,
                          ens_pred_multi_true=ens_pred_multi_true,
                          ens_pred_multi=ens_pred_multi)
        les_uncs_list = parallel_backend(delayed(process)(
            cc_mask=(y_pred_multi == cc_label).astype("float")
        ) for cc_label in cc_labels)  # returns lists of dictionaries, but need dictionary of lists

    if les_uncs_list:
        results = dict(zip(les_uncs_measures, [np.zeros_like(y_pred_multi, dtype='float') for _ in les_uncs_measures]))
        # for each measure of uncertainty create a lesion uncertainty map
        for cc_label, uncs_dict in zip(cc_labels, les_uncs_list):
            for unc_name in les_uncs_measures:
                results[unc_name] += uncs_dict[unc_name] * (y_pred_multi == cc_label).astype('float')
        return results
