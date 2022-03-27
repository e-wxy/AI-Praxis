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


### File List

模型训练需用到的文件

```
+---Data
|   +---ISIC2017
|   |   +---(ISIC-2017-Aug_Training_Data)     	# 原始扩充训练集 （提取patches时使用）
|   |   +---ISIC-2017_Test_Data
|   |   +---(ISIC-2017_Validation_Data)			# 原始验证集 （提取patches时使用）
|   |   +---ISIC-2017_Test_v2_Part3_GroundTruth.csv
|   |   +---ISIC-2017_Training_Aug_Part3_GroundTruth.csv
|   |   \---ISIC-2017_Validation_Part3_GroundTruth.csv
|   +---Training_Patch
|   \---Validation_Patch
+---model
|   |   +---densenet121-a639ec97.pth	# densenet （transfer learning 使用）
|   |   \---resnet50-19c8e357.pth		# resnet50 （ARL50预训练模型）
+---nets
|   |   +---__init__.py
|   |   +---arl.py
|   |   \---resnet.py			# baseline model
+---utils
|   |   +---__init__.py
|   |   +---data.py				# 数据加载
|   |   +---evaluation.py		# 模型评估
|   |   +---logger.py			# 日志
|   |   +---model.py			# 模型加载、存储，checkpoint
|   |   \---visualize.py		# plot 样本、混淆矩阵
+---ARL-CNN.ipynb				# ARL模型
\---(Transfer Learning.ipynb)	# run to test

```

预训练模型下载

```shell
# resnet50
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
# densenet121
wget https://download.pytorch.org/models/densenet121-a639ec97.pth
```



## Dataset

[ISIC Data](https://challenge.isic-archive.com/data/)

## 实验设置

1. Attention机制

   - [ ] 1-1 ARL-50
   - [ ] 1-2 ResNet-50

2. Loss Function

   ARL-14

   - Focal Loss
     - [ ] 2-1 alpha = [1, 1, 1]
     - [ ] 2-2 alpha = [average(class_nums)/num for num in class_nums]
   - Entropy Loss
     - [ ] 2-3 weights = [1, 1, 1]
     - [ ] 2-4 weights = [average(class_nums)/num for num in class_nums]

## Tips

Run in a terminal

```shell
# (activate env)
conda activate env_name
# convert .ipynb to .py
jupyter nbconvert --to script ARL-CNN.ipynb
# nohup
nohup python ARL-CNN.py
```

Check GPU info

```shell
nvidia-smi
```

