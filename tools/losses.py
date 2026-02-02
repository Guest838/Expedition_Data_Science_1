#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Refs : Multiclass losses by https://github.com/Nacriema/Loss-Functions-For-Semantic-Segmentation
which was adapted by us for multilabel purpose, and ours loss for binary classification
Reference papers: 
    * https://arxiv.org/pdf/2006.14822.pdf

"""
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, BCELoss
import torch.nn as nn
import torch.nn.functional as F

#OUR IMPLEMENTATION
class FocalDiceLossBinary(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, focal_weight=0.7, dice_weight=0.3):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    @staticmethod
    def focal_binary_loss(pred, target, alpha=0.75, gamma=2.0):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()

    @staticmethod
    def dice_binary_loss(pred, target, smooth=1e-6):
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        dice_coef = (2. * intersection + smooth) / (
            pred_sigmoid.sum() + target.sum() + smooth
        )
        return 1 - dice_coef

    def forward(self, pred, target):
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        target = target.float()
        focal_loss = self.focal_binary_loss(pred, target, self.alpha, self.gamma)
        dice_loss = self.dice_binary_loss(pred, target)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss


#FIXED FOR MULTILABEL
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


#FIXED FOR MULTILABEL
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


#FROM https://github.com/Alibaba-MIIL/ASL?ysclid=ml3ruf5bg6325778889
class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()

#FIXED FOR MULTILABEL
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
