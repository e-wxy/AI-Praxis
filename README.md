# AI Praxis Coursework

人工智能实践小组作业


## Environment

### Prerequisites

Package

```
pytorch
torchvision
numpy
matplotlib
pandas
seaborn
scikit-learn
opencv
# for segmentation tasks
segmentation-models-pytorch
albumentations
```

Create conda env

```shell
conda create --name env_name --file requirements.txt
conda activate env_name
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
pip install opencv-python
conda install ipykernel
python -m ipykernel install --user --name env_name --display-name "Python (env_name)"
conda install jupyter
```

Install packages for segmentation tasks

```shell
conda activate env_name
conda install segmentation-models-pytorch albumentations -c conda-forge --yes
```


### File List

模型训练需用到的文件

```
+---Data
|   +---ISIC2017
|   |   +---Aug_Training_Data			     	# 扩充训练集（原始+Extra）
|   |   +---ISIC-2017_Test_Data					# 原始测试集
|   |   +---ISIC-2017_Test_v2_Part1_GroundTruth		# 测试集masks
|   |   +---ISIC-2017_Training_Part1_GroundTruth	# 训练集masks
|   |   +---ISIC-2017_Validation_Data			# 原始验证集
|   |   +---ISIC-2017_Validation_Part1_GroundTruth	# 验证集masks
|   |   +---ISIC-2017_Test_v2_Part3_GroundTruth.csv
|   |   +---ISIC-2017_Training_Aug_Part3_GroundTruth.csv
|   |   \---ISIC-2017_Validation_Part3_GroundTruth.csv
+---model
|   |   \---resnet50-19c8e357.pth		# resnet50 （ARL50预训练模型）
+---nets
|   |   +---__init__.py
|   |   +---arl.py
|   |   \---resnet.py			# baseline model
+---utils
|   |   +---__init__.py
|   |   +---data.py			# 数据加载
|   |   +---evaluation.py		# 模型评估
|   |   +---logger.py			# 日志
|   |   +---loss.py			# 损失函数
|   |   +---model.py			# 模型加载、存储、训练，checkpoint
|   |   \---visualize.py		# plot 样本、混淆矩阵
+---ARL-CNN.ipynb			# ARL模型
+---Evaluation.ipynb			# 模型性能评估
+---Segmentation.ipynb			# 分割模型
\---seg_main.py				# 分割模型训练
```

预训练模型下载

```shell
# resnet50
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
```


## Dataset

[ISIC Data](https://challenge.isic-archive.com/data/)

分割任务mask

```
https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part1_GroundTruth.zip
https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Part1_GroundTruth.zip
https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Part1_GroundTruth.zip
```

## 实验设置

### Classification Task

1. Attention机制

   - [x] 1-1 ARL-50
   - [x] 1-2 ResNet-50

2. Loss Function

   ARL-14

   - Focal Loss
     - [x] 2-1 alpha = [1, 1, 1]
     - [x] 2-2 alpha = [2.5, 2.5, 0.5]
   - Entropy Loss
     - [x] 2-3 weights = [1, 1, 1]
     - [x] 2-4 weights = [2.5, 2.5, 0.5]

### Segmentation Task

| **实验编号** | **网络结构** | **预训练** | STATUS      |
| ------------ | ------------ | ---------- | ----------- |
| 1            | UNet         | 否         | In Progress |
| 2            | Dense-UNet   | 否         | In Progress |
| 3            | Dense-UNet   | 是         | Done        |
| 4            | Res-UNet     | 是         | In Progress |
| 5            | ARL-UNet     | 是         | In Progress |

## To Implement

- [x] 数据加载&增广
- [x] 网络搭建
- [x] 评价指标
  - [x] JSI/IoU
  - [x] DSC
  - [x] TJI
  - [x] SE & SP & ACC
- [ ] 分割效果可视化&比较
  - [ ] Seg-Evaluation.ipynb/py



## Tips

Run in a terminal

```shell
# (activate env)
conda activate env_name
# convert .ipynb to .py
jupyter nbconvert --to script ARL-CNN.ipynb
# nohup
nohup python ARL-CNN.py &
```

Run with argparse

```shell
## 分割实验2
python seg_main.py -c=0(or 1) -p=0
```

Check GPU info

```shell
nvidia-smi
```

Unzip

```shell
unzip ISIC-2017_Training_Part1_GroundTruth.zip
tar -xzvf Training_Patch.tar.gz
```