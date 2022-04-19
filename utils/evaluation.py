import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import pandas as pd
import cv2



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
def make_predictions(model, data_loader, device):
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
    

def accuracies(model, data_loader, device):
    """
    returns accuracy and balanced accuracy
    """
    y_true, y_pred = make_predictions(model, data_loader, device)
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


# AUC
@torch.no_grad()
def get_probs(model, data_loader, device):
    model.eval()
    label = []
    prob = [[1 for _ in range(3)]]
    soft = nn.Softmax(dim=-1)

    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        z = model(x)
        p = soft(z)
        prob = np.concatenate((prob, p.to('cpu')), axis=-2)
        label = np.concatenate((label, y.to('cpu')), axis=-1)
    
    prob = prob[1::]
    class_num = prob.shape[1]
    y = label_binarize(label, classes=[i for i in range(class_num)])

    return y, prob

def auc_scores(y, prob):
    class_num = prob.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Compute ROC curve and ROC area for each class
    for i in range(class_num):
        fpr[i], tpr[i], _ = metrics.roc_curve(y[:, i], prob[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area (computed globally)
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y.ravel(), prob.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area (simply average on each label)
    # aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(class_num)]))
    # interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(class_num):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # average it and compute AUC
    mean_tpr /= class_num

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc


class Evaluation:
    def __init__(self, device, categories, best_score=0) -> None:
        self.device = device
        self.categories = categories
        self.class_num = len(categories)
        self.best_score = best_score

    @torch.no_grad()
    def get_probs(self, model, data_loader):
        """ get predicted probabilities

        Returns:
            y: one-hot labels
            prob: predicted probabilities
        """
        model.eval()
        label = []
        prob = [[1 for _ in range(3)]]
        soft = nn.Softmax(dim=-1)

        for x, y in data_loader:
            x, y = x.to(self.device), y.to(self.device)
            z = model(x)
            p = soft(z)
            prob = np.concatenate((prob, p.to('cpu')), axis=-2)
            label = np.concatenate((label, y.to('cpu')), axis=-1)
        
        self.prob = prob[1::]
        self.class_num = prob.shape[1]
        self.label = label
        self.y = label_binarize(label, classes=[i for i in range(self.class_num)])

        return self.y, self.prob

    def get_preds(self):
        """ make prediction from probs

        Returns:
            np.arracy: predictions in index form
        """
        self.pred = np.argmax(self.prob, axis=1)
        return self.pred


    @torch.no_grad()
    def make_predictions(self, model, data_loader):
        """ make predictions on datasets

        Returns:
            lists of labels and predictions
        """
        model.eval()
        self.label = []
        self.pred = []

        for x, y in data_loader:
            x, y = x.to(self.device), y.to(self.device)
            z = model(x)
            _, yhat = torch.max(z.data, 1)
            self.label = np.concatenate((self.label, y.to('cpu')), axis=-1)
            self = np.concatenate((self.pred, yhat.to('cpu')), axis=-1)

        return self.label, self.pred

    def get_acc(self):
        self.acc = metrics.accuracy_score(self.label, self.pred)
        return self.acc

    def get_bacc(self):
        self.b_acc = metrics.balanced_accuracy_score(self.label, self.pred)
        return self.b_acc

    def get_f1(self):
        self.f1_score = list(metrics.f1_score(self.label, self.pred, average=None))
        return self.f1_score


    def binary_accuracies(self):
        """
        returns accuracies on MEL and SK in binary fashion
        """
        self.mel_pred = (self.pred == 0)
        self.mel_label = (self.label == 0)
        self.sk_pred = (self.pred == 1)
        self.sk_label = (self.label == 1)
        self.mel_acc = metrics.accuracy_score(self.mel_label, self.mel_pred)
        self.sk_acc = metrics.accuracy_score(self.sk_label, self.sk_pred)
        return self.mel_pred, self.sk_pred


    def auc_scores(self):
        self.fpr = dict()
        self.tpr = dict()
        self.roc_auc = dict()

        # Compute ROC curve and ROC area for each class
        for i in range(self.class_num):
            self.fpr[i], self.tpr[i], _ = metrics.roc_curve(self.y[:, i], self.prob[:, i])
            self.roc_auc[i] = metrics.auc(self.fpr[i], self.tpr[i])

        # Compute micro-average ROC curve and ROC area (computed globally)
        self.fpr["micro"], self.tpr["micro"], _ = metrics.roc_curve(self.y.ravel(), self.prob.ravel())
        self.roc_auc["micro"] = metrics.auc(self.fpr["micro"], self.tpr["micro"])

        # Compute macro-average ROC curve and ROC area (simply average on each label)
        # aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([self.fpr[i] for i in range(self.class_num)]))
        # interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.class_num):
            mean_tpr += np.interp(all_fpr, self.fpr[i], self.tpr[i])
        # average it and compute AUC
        mean_tpr /= self.class_num

        self.fpr["macro"] = all_fpr
        self.tpr["macro"] = mean_tpr
        self.roc_auc["macro"] = metrics.auc(self.fpr["macro"], self.tpr["macro"])

        return self.fpr, self.tpr, self.roc_auc

    def get_report(self):
        self.report = metrics.classification_report(self.label, self.pred, target_names=self.categories, digits=3)
        return self.report

    def get_confusion(self):
        """ calculate the confusion matrix

        Returns:
            DataFrame of confusion matrix: (i, j) - the number of samples with true label being i-th class and predicted label being j-th class.
        """
        c_matrix = metrics.confusion_matrix(self.label, self.pred)
        self.CMatrix = pd.DataFrame(c_matrix, columns=self.categories, index=self.categories)
        return self.CMatrix
    
    def complete_scores(self, mode="train"):
        """ compute evaluation scores from self.probs

        Args:
            mode (str): "train" or "test". Defaults to "train", in which, only acc and auc will be computed.
        """
        self.get_preds()
        self.get_acc()
        self.auc_scores()
        self.get_bacc()
        self.get_f1()

        if mode != "train":
            self.get_report()
            self.get_confusion()
            self.binary_accuracies()



    

# class activation mapping(CAM)

def CAM(feature_of_conv, weight_of_classifier, class_idxs,
        size_upsample=(224, 224)):
    """ calculate the class activation mapping

    Args:
        feature_of_conv(tensor of shape[bs, c, h, w]): the output of the last Conv layer
            just before the GAP(global average pooling) and FC(fully connected)
        weight_of_classifier(tensor of shape[num_of_classes, c]): the weight of the FC
        class_idx(int/list of int): the class index of the input image
        size_upsample(tuple): the output shape of CAM
    
    Returns:
        A list of 2d ndarray represents the CAM of the batch of inputs
    """
    bs, c, h, w = feature_of_conv.shape
    output_cams = []

    if type(class_idxs) == int:
        class_idxs = [class_idxs for _ in range(bs)]
    
    assert len(class_idxs) == bs, "the length of class_idxs not match the batch size."

    for i in range(bs):
        # compute cam
        weights = weight_of_classifier[class_idxs[i]].reshape((1, c))
        cam = weights @ feature_of_conv[i].reshape((c, h * w))
        cam = cam.reshape((h, w))

        # change to gray image
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())
        cam_img = np.uint8(255 * cam_img)
        output_cams.append(cv2.resize(cam_img, size_upsample))
    
    return output_cams
