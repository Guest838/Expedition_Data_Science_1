#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Nov 17 12:48:36 2021

@author: Nacriema

Refs:

I build the collection of loss that used in Segmentation Task, beside the Standard Loss provided by Pytorch, I also
implemented some loss that can be used to enhance the training process.

For me: Loss function is computed by comparing between probabilities, so in each Loss function if we pass logit as input
then we should convert them into probability. One-hot encoding also a form of probability.

For testing purpose, we should crete ideal probability for compare them. Then I give the loss function option use soft
max or not.

Maybe I need to convert each function inside the forward pass to the function that take the input and target as softmax
probability, inside the forward pass we just convert the logits into it


Should use each function, because most other functions like Exponential Logarithmic Loss use the result of the defined
function above for compute.

Difference between BCELoss and CrossEntropy Loss when consider with multiple classification (n_classes >= 3):
    - When I'm reading about the formula of CrossEntropy Loss for multiple class case, then I see the loss just
    "include" the t*ln(p) part, but not the (1 - t)ln(1 - p)
    for the "background" class. Then it can not "capture" the properties between each class with the background, just
    between each class together.
    - Then I'm reading from this thread https://github.com/ultralytics/yolov5/issues/5401, the author give me the same
    idea.


Reference papers: 
    * https://arxiv.org/pdf/2006.14822.pdf

"""
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, BCELoss
import torch.nn as nn
import torch.nn.functional as F  # noqa


# DONE #FIXED MULTILABEL
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0, reduction='none', eps=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, output, target):
        num_classes = output.shape[1]
        output_sigmoid = F.sigmoid(output)
        weight = torch.pow(1.0 - output_sigmoid , self.gamma)
        focal = -self.alpha * weight * output_sigmoid 
        # This line is very useful, must learn einsum, bellow line equivalent to the commented line
        # loss_tmp = torch.sum(focal.to(torch.float) * one_hot_target.to(torch.float), dim=1)
        loss_tmp = torch.einsum('bc..., bc...->b...', target, focal)
        if self.reduction == 'none':
            return loss_tmp
        elif self.reduction == 'mean':
            return torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            return torch.sum(loss_tmp)
        else:
            raise NotImplementedError(f"Invalid reduction mode: {self.reduction}")


# DONE #FIXED MULTILABEL
class TverskyLoss(nn.Module):
    """
    Tversky Loss is the generalization of Dice Loss
    It in the group of Region-Base Loss
    """
    def __init__(self, beta=0.5, use_sigmoid=True):
        """
        Args:
            beta:
        """
        super(TverskyLoss, self).__init__()
        self.beta = beta
        self.use_sigmoid = use_sigmoid

    def forward(self, output, target, epsilon=1e-6):
        num_classes = output.shape[1]
        if self.use_sigmoid:
            output = F.sigmoid(output)  # predicted value
        # Notice: TverskyIndex is numerator / denominator
        # See https://en.wikipedia.org/wiki/Tversky_index, we have the quick comparison between probability and set \
        # G is the Global Set, A_ = G - A, then
        # |A - B| = |A ^ B_| = |A ^ (G - B)| so |A - B| in set become (1 - target) * (output)
        # With ^ = *, G = 1
        numerator = torch.sum(output * target, dim=(-2, -1))
        denominator = numerator + self.beta * torch.sum((1 - target) * output, dim=(-2, -1)) + (1 - self.beta) * torch.sum(target * (1 - output), dim=(-2, -1))
        return 1 - torch.mean((numerator + epsilon) / (denominator + epsilon))


# DONE
class FocalTverskyLoss(nn.Module):
    """
    More information about this loss, see: https://arxiv.org/pdf/1810.07842.pdf
    This loss is similar to Tversky Loss, but with a small adjustment
    With input shape (batch, n_classes, h, w) then TI has shape [batch, n_classes]
    In their paper TI_c is the tensor w.r.t to n_classes index

    FTL = Sum_index_c(1 - TI_c)^gamma
    """
    def __init__(self, gamma=1, beta=0.5, use_sigmoid=True):
        super(FocalTverskyLoss, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.use_sigmoid = use_sigmoid

    def forward(self, output, target, epsilon=1e-6):
        num_classes = output.shape[1]
        if self.use_sigmoid:
            output = F.sigmoid(output)  # predicted value
        numerator = torch.sum(output * target, dim=(-2, -1))
        denominator = numerator + self.beta * torch.sum((1 - target) * output, dim=(-2, -1)) + (
                    1 - self.beta) * torch.sum(target * (1 - output), dim=(-2, -1))
        TI = torch.mean((numerator + epsilon) / (denominator + epsilon), dim=0)  # Shape [batch, num_classes], should reduce along batch dim
        return torch.sum(torch.pow(1.0 - TI, self.gamma))


# DONE #FIXED MULTILABEL
class ComboLoss(nn.Module):
    """
    It is defined as a weighted sum of Dice loss and a modified cross entropy. It attempts to leverage the 
    flexibility of Dice loss of class imbalance and at same time use cross-entropy for curve smoothing. 
    
    This loss will look like "batch bce-loss" when we consider all pixels flattened are predicted as correct or not

    Paper: https://arxiv.org/pdf/1805.02798.pdf. See the original paper at formula (3)
    Author's implementation in Keras : https://github.com/asgsaeid/ComboLoss/blob/master/combo_loss.py

    This loss is perfect loss when the training loss come to -0.5 (with the default config)
    """
    def __init__(self, use_sigmoid=True, ce_w=0.5, ce_d_w=0.5, eps=1e-12):
        super(ComboLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.ce_w = ce_w
        self.ce_d_w = ce_d_w
        self.eps = 1e-12
        self.smooth = 1

    def forward(self, output, target):
        num_classes = output.shape[1]
        
        # Apply softmax to the output to present it in probability.
        if self.use_sigmoid:
            output = F.sigmoid(output)
        
        
        # At this time, the output and one_hot_target have the same shape        
        y_true_f = torch.flatten(target)
        y_pred_f = torch.flatten(output)
        intersection = torch.sum(y_true_f * y_pred_f)
        d = (2. * intersection + self.smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + self.smooth)

        # From this thread: https://discuss.pytorch.org/t/bceloss-how-log-compute-log-0/11390. Use this trick to avoid
        # nan when log(0) and log(1)
        out = - (self.ce_w * y_true_f * torch.log(y_pred_f + self.eps) + (1 - self.ce_w) * (1.0 - y_true_f) * torch.log(1.0 - y_pred_f + self.eps))
        weighted_ce = torch.mean(out, axis=-1)

        # Due to this is the hybrid loss, then the loss can become negative:
        # https://discuss.pytorch.org/t/negative-value-in-my-loss-function/101776
        combo = (self.ce_d_w * weighted_ce) - ((1 - self.ce_d_w) * d)
        return combo
