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
    """
    Calculate the Dice Loss between `pred` and `target`.

    Supports:
    - V-Net style Dice Loss (squared denominator)
    - Naive Dice Loss (linear denominator)

    Args:
        pred (Tensor): Predicted probabilities, shape (N, *).
        target (Tensor): Ground truth masks, shape (N, *), same shape as `pred`.
        weight (Tensor, optional): Optional weight for each instance. Shape: (N,).
        eps (float): Small value to avoid division by zero. Default: 1e-3.
        reduction (str): Reduction method - 'none', 'mean', 'sum'. Default: 'mean'.
        naive_dice (bool): Use naive Dice (linear denominator) if True. Default: False.
        avg_factor (int, optional): Normalization factor. Default: None.

    Returns:
        Tensor: Dice loss.
    """
    # Flatten input and target to (N, -1)
    input = pred.flatten(1)

    # Resize target to match pred if necessary
    if target.shape != pred.shape:
        target = torch.nn.functional.interpolate(
            target.unsqueeze(1).float(),  # Add channel dimension
            size=pred.shape[2:],          # Match spatial dims (H, W)
            mode='bilinear',              # or 'nearest' if binary masks
            align_corners=False
        ).squeeze(1)  # Remove channel dimension after interpolation

    target = target.flatten(1).float()

    # Intersection
    intersection = torch.sum(pred * target, dim=1)

    if naive_dice:
        union = torch.sum(pred, dim=1) + torch.sum(target, dim=1)
    else:
        union = torch.sum(pred * pred, dim=1) + torch.sum(target * target, dim=1)

    # Dice coefficient
    dice_score = (2 * intersection + eps) / (union + eps)
    loss = 1 - dice_score

    # Apply weight if provided and dimensions match
    if weight is not None:
        if weight.shape[0] == loss.shape[0]:
            loss = loss * weight
        else:
            raise ValueError(f"Weight shape {weight.shape} does not match loss shape {loss.shape}")

    # Reduce loss
    loss = weight_reduce_loss(loss, weight=None, reduction=reduction, avg_factor=avg_factor)
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
