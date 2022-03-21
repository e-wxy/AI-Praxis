import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np
from sklearn import metrics
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# loss function

class FocalLoss(nn.Module):
    def __init__(self, alpha: list, gamma=2, num_classes: int = 10, reduction='mean'):
        """ Focal Loss

        Args:
            alpha (list): 类别权重 class weight
            gamma (int/float): 难易样本调节参数 focusing parameter
            num_classes (int): 类别数量 number of classes
            reduction (string): 'mean', 'sum', 'none'
        """
        super(FocalLoss, self).__init__()
        assert len(alpha) == num_classes, "alpha size doesn't match with class number"
        self.alpha = torch.Tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds, labels):
        """
        Shape:
            preds: [B, N, C] or [B, C]
            labels: [B, N] or [B]
        """
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1)
        preds_logsoft = torch.log(preds_softmax)

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = - torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft) # torch.pow((1-preds_softmax), self.gamma) - (1-pt)**γ
        loss = torch.mul(self.alpha, loss.t())
        
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

    def __str__(self) -> str:
        return "Criterion: Focal Loss\n α = {}\n γ = {}".format(self.alpha, self.gamma)



# predict

def predict(model, x):
    model.eval()
    with torch.no_grad():
        z = model(x)
        _, pred = torch.max(z.data, 1)
    return pred


@torch.no_grad()
def make_predictions(model, data_loader):
    """ make predictions on datasets

    Returns:
        lists of labels and predictions
    """
    model.eval()
    label = []
    pred = []

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        z = model(x)
        _, yhat = torch.max(z.data, 1)
        label = np.concatenate((label, y.to('cpu')), axis=-1)
        pred = np.concatenate((pred, yhat.to('cpu')), axis=-1)

    return label, pred
    

def accuracies(model, data_loader):
    """
    returns accuracy and balanced accuracy
    """
    y_true, y_pred = make_predictions(model, data_loader)
    acc = metrics.accuracy_score(y_true, y_pred)
    b_acc = metrics.balanced_accuracy_score(y_true, y_pred)
    f1_score = metrics.f1_score(y_true, y_pred, average=None)

    return acc, b_acc, f1_score


# confusion matrix

def get_confusion(y_true, y_pred, categories):
    """ calculate the confusion matrix

    Returns:
        DataFrame of confusion matrix: (i, j) - the number of samples with true label being i-th class and predicted label being j-th class.
    """
    c_matrix = metrics.confusion_matrix(y_true, y_pred)
    CMatrix = pd.DataFrame(c_matrix, columns=categories, index=categories)
    return CMatrix