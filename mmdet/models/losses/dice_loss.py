# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmdet.registry import MODELS
from .utils import weight_reduce_loss


def dice_loss(pred,
              target,
              weight=None,
              eps=1e-3,
              reduction='mean',
              naive_dice=False,
              avg_factor=None):
    # Ensure pred and target have the same spatial shape
    if pred.dim() == 4 and target.dim() == 3:
        # Add channel dimension to target
        target = target.unsqueeze(1)
    elif pred.dim() == 3 and target.dim() == 4:
        pred = pred.unsqueeze(1)
    # Now both should be [N, 1, H, W] or [N, C, H, W]
    if pred.shape[-2:] != target.shape[-2:]:
        pred = torch.nn.functional.interpolate(
            pred, size=target.shape[-2:], mode='bilinear', align_corners=False
        )
    # Ensure channel dims match
    if pred.shape[1] != target.shape[1]:
        if target.shape[1] == 1 and pred.shape[1] > 1:
            target = target.expand(pred.shape[0], pred.shape[1], *target.shape[2:])
        elif pred.shape[1] == 1 and target.shape[1] > 1:
            pred = pred.expand(target.shape[0], target.shape[1], *pred.shape[2:])
    input = pred.flatten(1)
    target = target.flatten(1).float()

    a = torch.sum(input * target, 1)
    if naive_dice:
        b = torch.sum(input, 1)
        c = torch.sum(target, 1)
        d = (2 * a + eps) / (b + c + eps)
    else:
        b = torch.sum(input * input, 1) + eps
        c = torch.sum(target * target, 1) + eps
        d = (2 * a) / (b + c)

    loss = 1 - d
    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@MODELS.register_module()
class DiceLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 activate=True,
                 reduction='mean',
                 naive_dice=False,
                 loss_weight=1.0,
                 eps=1e-3):
        """Compute dice loss.

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            activate (bool): Whether to activate the predictions inside,
                this will disable the inside sigmoid operation.
                Defaults to True.
            reduction (str, optional): The method used
                to reduce the loss. Options are "none",
                "mean" and "sum". Defaults to 'mean'.
            naive_dice (bool, optional): If false, use the dice
                loss defined in the V-Net paper, otherwise, use the
                naive dice loss in which the power of the number in the
                denominator is the first power instead of the second
                power. Defaults to False.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            eps (float): Avoid dividing by zero. Defaults to 1e-3.
        """

        super(DiceLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.naive_dice = naive_dice
        self.loss_weight = loss_weight
        self.eps = eps
        self.activate = activate

    def forward(self,
                pred,
                target,
                weight=None,
                reduction_override=None,
                avg_factor=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction, has a shape (n, *).
            target (torch.Tensor): The label of the prediction,
                shape (n, *), same shape of pred.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction, has a shape (n,). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.activate:
            if self.use_sigmoid:
                pred = pred.sigmoid()
            else:
                raise NotImplementedError

        loss = self.loss_weight * dice_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            naive_dice=self.naive_dice,
            avg_factor=avg_factor)

        return loss