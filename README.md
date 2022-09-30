# Multi-scale comparative study of uncertainty measures for white matter lesion segmentation

This repository contains the code used for the study of voxel- and lesion- scale uncertainty in application to 
white matter multiple sclerosis segmentation task. It is supplementary to our [paper]().

The code includes:
* Implementations of voxel-scale uncertainty measures, such as mutual information (MI), expected pair-wise KL divergence (EPKL) and reverse mutual information (RMI) for knowledge uncertainty; expected entropy (ExE) for data uncertainty; entropy of expected (EoE) and negated confidence (NC).
* Implementations of lesion-scale uncertainty measures, such as mean or logsum of voxels' uncertainties within the lesion region, proposed _Detection disagreement uncertainty_ (DDU).
* Implementation of the retention curves on the voxel and lesion scales.

Requirements
---

Code wads compiled with python 3.8. All the requirements are listed in "requirements.txt" and can be installed using a pip command:

```commandline
pip3 install -r requirements.txt
```

Model
----

The model used for the current study is a shallow U-net architecture. The code for training and testing was developed during the [Shifts project](https://shifts.ai/) and is avalable in the [Shifts GitHub](https://github.com/Shifts-Project/shifts/tree/main/mswml).

Data
----

The data was also provided by the Shifts project. Already preprocessed data can be downloaded from [zenodo](https://zenodo.org/record/7051658) under the OFSEP data usage agreement. Data is distributed under CC BY NC SA 4.0 license.

The "train" and "dev in" sets were used for training and validation. The "eval in" and "dev out" sets were used for retention curves construction.

Code organisation
----

The code contains several modules with aforementioned implementations. All the functions inside the modules contain docstring documentation.

### Module "voxel_uncertainty_measures.py"

Contains implementations of voxel-scale uncertainty measures. Function
`ensemble_uncertainties_classification(probs, epsilon) -> dict` allows obtaining 
all the uncertainty maps for a single scan described in the paper.

### Module "lesion_uncertainty_measures.py"

Contains implementations of lesion-scale uncertainty measures described in the paper. Function 
`lesions_uncertainty(y_pred_multi, vox_unc_maps, ens_pred_multi, ens_pred_multi_true, n_jobs, dl)` allows obtaining
lesion uncertainty measures for each lesion in the scan. Function `lesions_uncertainty_maps(y_pred_multi, vox_unc_maps, ens_pred_multi, ens_pred_multi_true, n_jobs, dl) -> dict` allows obtaining
different lesion uncertainty maps for a particular scans.

### Module "error_retention_curves.py"

Contains implementations of voxel- and lesion- scale retention curves.
Function `voxel_scale_rc(y_pred, y, uncertainties, fracs_retained, metric_name, n_jobs) -> tuple` given uncertainty estimates builds voxel scale retention curve for a chosen segmentation quality metric and computes area above the retention curve for a single scan.
Function `lesion_scale_rc_nofn(lesion_uncertainties, lesion_types, n_fn, fracs_retained) -> tuple` given uncertainty estimates builds lesion F1 retention curve for a particular scan, it does not remove false negative lesions (used in the paper).
Function `lesion_scale_rc(--"--)` does the same, but excludes false negative lesions, thus provides a more theoretical evaluation of an uncertainty measure.

### Module "lesion_extraction.py"

Contains a function to extract TP, FP and FN lesions `get_tp_fp_fn_lesions(...)` given predicted segmentation map and a ground truth, 
as well as a function to count the number of FN lesions `gt_fn_count(...)`.

Citation
---

In case of using this code, please cite:

```
```
