# MS_WML_uncs
This repository contains the code used for the study of voxel- and lesion- scale uncertainty in application to 
white matter multiple sclerosis segmentation task. 

The code includes:
* Implementations of voxel-scale uncertainty measures, such as mutual information (MI), expected pair-wise KL divergence (EPKL) and reverse mutual information (RMI) for knowledge uncertainty; expected entropy (ExE) for data uncertainty; entropy of expected (EoE) and negated confidence (NC).
* Implementations of lesion-scale uncertainty measures, such as mean or logsum of voxels' uncertainties within the lesion region, proposed _Detection disagreement uncertainty_ (DDU).
* Implementation of the retention curves on the voxel and lesion scales.

Model
----

The model used for the current study is a shallow U-net architecture. The code for training and testing was developed during the [Shifts project](https://shifts.ai/) and is avalable in the [Shifts GitHub](https://github.com/Shifts-Project/shifts/tree/main/mswml).

Data
----

The data was also provided by the Shifts project. Already preprocessed data can be downloaded from [zenodo](https://zenodo.org/record/7051658) under the OFSEP data usage agreement. Data is distributed under CC BY NC SA 4.0 license.

The "train" and "dev in" sets were used for training and validation. The "eval in" and "dev out" sets were used for retention curves construction.

Code organisation
----

