import torch
import segmentation_models_pytorch as smp
import nets
from torch.utils import data
from torchvision import transforms
import albumentations as A
import numpy as np
import os
import argparse
import utils
import warnings
warnings.filterwarnings('ignore')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16
NUM_WORKERS = 0 
WIDTH = 336  # 512
HEIGHT = 224 # 384


# Command-Line Parsing
parser = argparse.ArgumentParser(description='Semantic Segmentation on ISIC 2017')
parser.add_argument('-i', '--id', type=int, help='model id')
parser.add_argument('-n', '--net', type=str, default='dense', help="Network architecture (default: 'dense')")
parser.add_argument('-c', '--checked', type=int, default=1, choices=[0, 1], help='Whether continuing from checkpoint (default: 1(True))')
parser.add_argument('-p', '--pretrained', type=int, default=1, choices=[0, 1], help='Whether using pretrained model (default: 1(True))')
parser.add_argument('-t', '--title', type=str, default='', help='Commit messages on the experiment')

args = parser.parse_args()


# Log
model_id = args.id
title = 'unet'
model_name = title + '_' + str(model_id)
log = utils.Logger(verbose=True, title=os.path.join('seg', title))
log.logger.info("{} | Network: {}, Pretrained: {} | ".format(model_name, args.net, args.pretrained, args.title))


# Data Preparation

pth_train_img = 'Data/ISIC2017/Aug_Training_Data'
pth_valid_img = 'Data/ISIC2017/ISIC-2017_Validation_Data'
pth_test_img = 'Data/ISIC2017/ISIC-2017_Test_v2_Data'
pth_train_mask = 'Data/ISIC2017/ISIC-2017_Training_Part1_GroundTruth'
pth_valid_mask = 'Data/ISIC2017/ISIC-2017_Validation_Part1_GroundTruth'
pth_test_mask = 'Data/ISIC2017/ISIC-2017_Test_v2_Part1_GroundTruth'

ann_train = 'Data/ISIC2017/ISIC-2017_Training_Part3_GroundTruth.csv'
ann_valid = 'Data/ISIC2017/ISIC-2017_Validation_Part3_GroundTruth.csv'
ann_test = 'Data/ISIC2017/ISIC-2017_Test_v2_Part3_GroundTruth.csv'


# # augmentation transforms for both images and masks
trans_train = A.Compose([# A.ElasticTransform(),
                         A.RandomResizedCrop(width=WIDTH, height=HEIGHT, scale=(0.6, 1.3), ratio=(0.75, 1.3333333333333333)),
                         A.Flip(p=0.5),
                         A.Rotate(limit=180),
                         # A.Sharpen(),
                         # A.ColorJitter(),
                         # A.GaussNoise(),
                         ])

trans_test = A.Compose([A.Resize(height=int(HEIGHT*1.1), width=int(WIDTH*1.1)),
                        A.CenterCrop(height=HEIGHT, width=WIDTH)
                        ])

# # normalization
trans_img = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])
                                ])

def trans_mask(mask):
    return torch.as_tensor(np.array(mask/255), dtype=torch.int64)

# # dataloader
train_data = utils.SegData(ann_train, pth_train_img, pth_train_mask, trans_train, trans_img, trans_mask)
valid_data = utils.SegData(ann_valid, pth_valid_img, pth_valid_mask, trans_test, trans_img, trans_mask)

train_loader = data.DataLoader(train_data, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
valid_loader = data.DataLoader(valid_data, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


# Model Design

encoder = args.net
decoder_channels = [512, 256, 128, 64]

if encoder == 'dense': 
    model = smp.Unet('densenet121', encoder_weights='imagenet', classes=2, activation=None, encoder_depth=len(decoder_channels), decoder_channels=decoder_channels)
elif encoder == 'res':
    model = smp.Unet('resnet18', encoder_weights='imagenet', classes=2, activation=None, encoder_depth=len(decoder_channels), decoder_channels=decoder_channels)
elif encoder == 'unet':
    model = nets.UNet(3, 2)
else:
    raise ValueError('Not Implemented')
    
model = model.to(device)
log.logger.info("Encoder: {} | Decoder Channels: {} | Size: ({}, {})".format(encoder, decoder_channels, WIDTH, HEIGHT))

# Training
init_lr = 1e-4
weight_decay = 1e-4
max_epoch = 150
test_period = 1
early_threshold = 45

criterion = utils.DiceCE()
optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=0)

log.logger.info("Criterion: {}\nOptimizer: {}\nScheduler: {}".format(criterion, optimizer, scheduler))

if args.checked:
    trainer = utils.SegTrain(device, log, model_name, optimizer, scheduler, 0, 0, model)
else: 
    trainer = utils.SegTrain(device, log, model_name, optimizer, scheduler, 0, 0, None)

acc, iou = trainer.eval(model, valid_loader)
log.logger.info("Initial Performance on Valid Set: Acc: {}, IoU: {}".format(acc, iou))

history = trainer.fit(model, train_loader, valid_loader, criterion, max_epoch, test_period, early_threshold)



# Basic Evaluation

from utils.evaluation import mIoU, pixel_accuracy

@torch.no_grad()
def seg_predict(model, data_loader):
    model.eval()
    label = None
    pred = None

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        z = model(x)
        yhat = z.data
        if label == None:
            label = y.to('cpu')
            pred = yhat.to('cpu')
        else:
            label = torch.cat((label, y.to('cpu')), 0)
            pred = torch.cat((pred, yhat.to('cpu')), 0)

    return label, pred
    

del train_loader, valid_loader

test_data = utils.SegData(ann_test, pth_test_img, pth_test_mask, trans_test, trans_img, trans_mask)
test_loader = data.DataLoader(test_data, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

mask, pred_mask = seg_predict(model, test_loader)

pixel_acc = pixel_accuracy(pred_mask, mask)
log.logger.info("Pixel Accuracy: {}".format(pixel_acc))

iou_score = mIoU(pred_mask, mask)
log.logger.info("Mean IoU: {}".format(iou_score))


