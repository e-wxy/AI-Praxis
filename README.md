# AI Praxis Coursework

人工智能实践小组作业


## Environment
[SageMaker Studio Lab](https://studiolab.sagemaker.aws/)

### Prerequisites

```
pytorch
torchvision
numpy
matplotlib
pandas
seaborn
scikit-learn
```

### File List

模型训练需用到的文件

```
+---Data
|   +---ISIC2017
|   |   +---ISIC-2017-Aug_Training_Data
|   |   +---ISIC-2017_Test_Data
|   |   +---ISIC-2017_Validation_Data
|   |   +---ISIC-2017_Test_v2_Part3_GroundTruth.csv
|   |   +---ISIC-2017_Training_Aug_Part3_GroundTruth.csv
|   |   \---ISIC-2017_Validation_Part3_GroundTruth.csv
|   \---Training_Patch
+---utils
|   |   +---__init__.py
|   |   +---data.py				# 数据加载
|   |   +---evaluation.py		# 模型评估
|   |   +---logger.py			# 日志
|   |   +---model.py			# 模型加载、存储，checkpoint
|   |   \---visualize.py		# plot 样本、混淆矩阵
\---Transfer Learning.ipynb		# run to test

```



## Dataset

[ISIC Data](https://challenge.isic-archive.com/data/)
