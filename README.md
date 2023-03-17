# ASTL
An active semi-supervised transfer learning method for robot pose error prediction and compensation

## Contribution
- 1.The robot pose error prediction is defined as a transfer learning paradigm for the first time. Measurement-based and calibration-based methods for pose error perception are focused on with a new perspective.

- 2.The multi-stage greedy sampling (MGS)strategy is proposed to achieve informed selection and measurement of a few samples.The semi-supervised transfer learning (STL) is proposed to achieve the transfer of domain knowledge in the form of model,data and loss function from the simulation domain to the measurement domain.

- 3.An active semi-supervised transfer learning method (ASTL) for robot pose error prediction is proposed integrating the MGS and the STL,which achieve efficient and accurate prediction of robot pose error.

## Structure
An active semi-supervised transfer learning method (ASTL) is proposed by integrating the multi-stage greedy sampling (MGS) and semi-supervised transfer learning (STL).The MGS is used for the selection and labeling of few samples,and the STL is used for the prediction of unlabeled parts.
<div align=center>
<img src=https://github.com/ZhangTeng-Hust/ASTL/blob/main/IMG/all.png>
</div>

## Results
The relationship between the predicted and actual values of each dimension is represented as follows.
<div align=center>
<img src=https://github.com/ZhangTeng-Hust/ASTL/blob/main/IMG/result1.png>
</div>

The results of the frequency histograms of the position and orientation errors before and after compensation are compared as follows.
<div align=center>
<img src=https://github.com/ZhangTeng-Hust/ASTL/blob/main/IMG/Result2.png>
<img src=https://github.com/ZhangTeng-Hust/ASTL/blob/main/IMG/Result3.png>
</div>

## Special Reminder
No reproduction without permission！！！

This work is currently being submitted to Elsevier, so no individual or organization can fork this repository and ues the data involved in this repository until it has been reviewed.

## Citation Notes
If you would like to use some or all of the data in this repository, please cite our dataset and we will also be adding a citation format for articles subsequently.
### Dataset
Zhang, Teng; Peng, Fangyu; Tang, Xiaowei; Yan, Rong; Zhang, Chi; Deng, Runpeng (2023), “HUST_NC_Robot”, Mendeley Data, V1, doi: 10.17632/srxktn4752.1 (https://data.mendeley.com/datasets/srxktn4752)
