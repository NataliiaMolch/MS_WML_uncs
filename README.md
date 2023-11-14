# Multiple Sclerosis White Matter Lesion Uncertainty Measures At Multiple Anatomical scales

![Graphical abstract](graphical_abstract.png)

This repository contains the code used for the study of voxel-, lesion-, and patient- scale uncertainty in application to white matter multiple sclerosis segmentation task. For more details, please consult the manuscripts.

The code includes:
* Implementations of voxel-scale uncertainty measures, such as mutual information (MI), expected pair-wise KL divergence (EPKL) and reverse mutual information (RMI) for knowledge uncertainty; expected entropy (ExE) for data uncertainty; entropy of expected (EoE) and negated confidence (NC).
* Implementations of lesion-scale uncertainty measures, such as mean or logsum of voxels' uncertainties within the lesion region, proposed lesion structural uncertainty (LSU and LSU$^{+}$).
* Implementation of the patient-scale uncertainty measures either based on the aggregation from voxel or lesion scales or based on the disagreement in structural predictions (proposed PSU and PSU$^+$).
* Implementation of the retention curves at each of the anatomica scales.

Model
----

The model used for the current study is a U-net architecture. The code for training and testing as well as the model weights was developed during the [Shifts project](https://shifts.ai/) and is avalable in the [Shifts GitHub](https://github.com/Shifts-Project/shifts/tree/main/mswml).

Data
----

The data was also provided by the Shifts project. Already preprocessed data can be downloaded from [zenodo](https://zenodo.org/record/7051658) under the OFSEP data usage agreement. Data is distributed under CC BY NC SA 4.0 license.

The "train" and "dev in" sets were used for training and validation. The "eval in" and "dev out" sets were used for retention curves construction.

Citiation
----

Cite these works in case of using the code.

```bibtex
@INPROCEEDINGS{10230563,
  author={Molchanova, Nataliia and Raina, Vatsal and Malinin, Andrey and La Rosa, Francesco and Muller, Henning and Gales, Mark and Granziera, Cristina and Graziani, Mara and Cuadra, Meritxell Bach},
  booktitle={2023 IEEE 20th International Symposium on Biomedical Imaging (ISBI)}, 
  title={Novel Structural-Scale Uncertainty Measures and Error Retention Curves: Application to Multiple Sclerosis}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ISBI53787.2023.10230563}}
```