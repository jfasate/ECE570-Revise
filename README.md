# ECE570-Single-Stream-CNN_Revise# 

CNNs for Multi-Source Remote Sensing Data Fusion

## Description

Pytorch implementation of the paper "Single-stream CNN with Learnable Architecture for Multi-source Remote Sensing Data". 

The original paper can be found below:

Find our paper at: [[IEEE Xplore]](https://ieeexplore.ieee.org/document/9761218)  [[arxiv]](http://arxiv.org/abs/2109.06094)

## Usage
It is highly advisable to install the below requirement manually. As it work perfectly in my computer it might not work perfectly in every computer.

- Requirements: python3, pytorch, gdal, sklearn.

## Data
Data files are available at [this Google Drive site](https://drive.google.com/drive/folders/1urY6Pjba3mStDcRphIfkNf50295aW2o2?usp=sharing), which can be directly used in this code.

## Run
Here in this repository there is no data folder, please create the data folder and as you must have downloaded the data file from above, please take all the .mat file from folder and paste them directly inside the data folder. Before runing. Please set the correct path in common.py as guided below:

In Config class of ``` common.py``` please set path as:

save_ckpt_dir = '/path/to/train_model'

result_out_dir = '/path/to/results'

data_dir = '/path/to/data'

You gonna find all the train_model, results, data folder inside repository.

- Simply run 
```
python3 main.py
```
- To customize training/model arguments, modify ```common.py```. Arguments are automatically loaded to ```main.py```.

Please contact if found any issue.

## Baseline models

This repository also contains Pytorch implementation of the following models, which we use as baselines: 

- _Fusion-FCN_: A three-branch CNN for MS-HSI-LiDAR data fusion. Award-winning model in 2018 IEEE DFC. 
[[Paper]](https://ieeexplore.ieee.org/abstract/document/8518295/): "Multi-Source Remote Sensing Data Classification via Fully Convolutional Networks and Post-Classification Processing"

- _Two-branch CNN_ (_TB-CNN_): A two-branch CNN architecture for feasture fusion with HSI and other remote scensing imagery. [[Paper]](https://ieeexplore.ieee.org/abstract/document/8068943): "Multisource Remote Sensing Data Classification Based on Convolutional Neural Network" [[Official Tensorflow implementation]](https://github.com/Hsuxu/Two-branch-CNN-Multisource-RS-classification)

Implementation of these models can be found at ```model/baseline/```. 


## Citation

If you find our work helpful, please kindly cite: 
```
@ARTICLE{9761218,
  author={Yang, Yi and Zhu, Daoye and Qu, Tengteng and Wang, Qiangyu and Ren, Fuhu and Cheng, Chengqi},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Single-Stream CNN With Learnable Architecture for Multisource Remote Sensing Data}, 
  year={2022},
  volume={60},
  number={},
  pages={1-18},
  doi={10.1109/TGRS.2022.3169163}}
```
