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
|   |   +---arl18.pth				# arl18（ARL-UNet预训练参数）
|   |   \---resnet50-19c8e357.pth		# resnet50 （ARL50预训练参数）
+---nets
|   |   +---__init__.py
|   |   +---arl.py
|   |   +---arl_unet.py
|   |   +---res_unet.py
|   |   +---resnet.py
|   |   \---unet.py
+---utils
|   |   +---__init__.py
|   |   +---data.py			# 数据加载
|   |   +---evaluation.py		# 模型评估
|   |   +---logger.py			# 日志
|   |   +---loss.py			# 损失函数
|   |   +---model.py			# 模型加载、存储、训练，checkpoint
|   |   \---visualize.py		# plot 样本、混淆矩阵
+---ARL-CNN.ipynb			# 分类模型
+---Evaluation-Cls.ipynb			# 分类模型性能评估
+---Evaluation-Seg.ipynb			# 分割模型性能评估
+---Segmentation.ipynb			# 分割模型
\---seg_main.py				# 分割模型训练
```

预训练模型下载

```shell
# resnet50
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
```


## Dataset

ISIC 2017: [ISIC Data](https://challenge.isic-archive.com/data/)

分割任务mask

```
https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part1_GroundTruth.zip
https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Validation_Part1_GroundTruth.zip
https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Part1_GroundTruth.zip
```

## Experimental Settings and Results

### Classification Task

An unofficial implementation of *Attention Residual Learning for Skin Lesion Classification*[^1].

1. Attention Mechanism

   - [x] 1-1 ARL-50
   - [x] 1-2 ResNet-50

   | Network Architecture  | Melanoma  |           |             |             | Seborrheic  Keratosis |           |             |             |
   | --------------------- | :-------: | :-------: | :---------: | :---------: | :-------------------: | :-------: | :---------: | :---------: |
   |                       |    AUC    |    ACC    | Sensitivity | Specificity |          AUC          |    ACC    | Sensitivity | Specificity |
   | ARL-CNN14             | **0.882** | **0.857** |  **0.632**  |    0.911    |         0.955         |   0.925   |  **0.811**  |    0.945    |
   | ARL-CNN50             |   0.861   |   0.853   |    0.590    |    0.917    |       **0.959**       | **0.927** |    0.722    |    0.963    |
   | Baseline  (ARL-CNN50) |   0.875   |   0.850   |    0.658    |    0.896    |         0.958         |   0.868   |    0.878    |    0.867    |
   | ResNet50              |   0.849   | **0.860** |    0.504    |  **0.946**  |         0.919         |   0.908   |    0.533    |  **0.975**  |
   | Baseline  (ResNet50)  |   0.857   |   0.838   |    0.632    |    0.888    |         0.948         |   0.842   |    0.867    |    0.837    |

   

2. Loss Function

   Network: ARL-14

   - Focal Loss
     - [x] 2-1 alpha = [1, 1, 1]
     - [x] 2-2 alpha = [2.5, 2.5, 0.5]
   - Entropy Loss
     - [x] 2-3 weights = [1, 1, 1]
     - [x] 2-4 weights = [2.5, 2.5, 0.5]

   |          ID          |   Loss  Function   |  Class Weights   | Melanoma  |           |             |             | Seborrheic  Keratosis |           |             |             |
   | :------------------: | :----------------: | :--------------: | :-------: | :-------: | :---------: | :---------: | :-------------------: | :-------: | :---------: | :---------: |
   |                      |                    |                  |    AUC    |    ACC    | Sensitivity | Specificity |          AUC          |    ACC    | Sensitivity | Specificity |
   |         2-1          |    Focal  Loss     |    [1,  1, 1]    |   0.846   |   0.858   |    0.538    |  **0.936**  |         0.924         |   0.912   |    0.511    |    0.982    |
   |         2-2          |                    | [2.5,  2.5, 0.5] |   0.863   | **0.862** |    0.556    |  **0.936**  |         0.927         |   0.903   |    0.444    |  **0.984**  |
   |         2-3*         | CrossEntropy  Loss |    [1,  1, 1]    |   0.822   |   0.835   |    0.487    |    0.919    |         0.929         |   0.893   |  **0.833**  |    0.904    |
   |         2-4*         |                    | [2.5,  2.5, 0.5] | **0.882** |   0.857   |  **0.632**  |    0.911    |         0.955         | **0.925** |    0.811    |    0.945    |
   | Baseline (ARL-CNN50) |                    |                  |   0.875   |   0.850   |    0.658    |    0.896    |       **0.958**       |   0.868   |  **0.878**  |    0.867    |

`*: ran on RTX 3090, while others ran on A30.

### Segmentation Task

Semantic segmentation on ISIC 2017 based on *U-Net: Convolutional Networks for Biomedical Image Segmentation*[^2].

The down-sampling conv layers are changed into blocks in experiment 2~5.

|  ID  | Settings             |            |  Results   |            |            |            |            |            |
| :--: | -------------------- | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
|      | Network Architecture | Pretrained |    ACC     |     SE     |     SP     |    JSI     |    DSC     |    TJI     |
|  1   | UNet                 |     F      |   0.8924   |   0.6872   |   0.9705   |   0.7525   |   0.8539   |   0.6784   |
|  2   | Dense-UNet           |     F      |   0.9215   |   0.8224   |   0.9593   |   0.8207   |   0.8995   |   0.7420   |
|  3   | Dense-UNet           |     T      | **0.9335** | **0.8452** |   0.9672   | **0.8457** | **0.9150** | **0.7875** |
|  4   | Res-UNet             |     T      |   0.9222   |   0.7765   | **0.9778** |   0.8174   |   0.8972   |   0.7447   |
|  5   | ARL-UNet             |     T      |   0.9220   |   0.7795   |   0.9764   |   0.8173   |   0.8972   |   0.7413   |



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



## References

[^1]: Zhang, Jianpeng, Yutong Xie, Yong Xia, and Chunhua Shen. ‘Attention Residual Learning for Skin Lesion Classification’. *IEEE Transactions on Medical Imaging* 38, no. 9 (September 2019): 2092–2103. https://doi.org/10.1109/TMI.2019.2893944.
[^2]: Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. ‘U-Net: Convolutional Networks for Biomedical Image Segmentation’. In *Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015*, edited by Nassir Navab, Joachim Hornegger, William M. Wells, and Alejandro F. Frangi, 9351:234–41. Lecture Notes in Computer Science. Cham: Springer International Publishing, 2015. https://doi.org/10.1007/978-3-319-24574-4_28.