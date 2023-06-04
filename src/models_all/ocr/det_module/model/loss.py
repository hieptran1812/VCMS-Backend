import torch
import torch.nn as nn
import torch.nn.functional as F


class OHEMBalanceCrossEntropyLoss(nn.Module):
    def __init__(self, negative_ratio=3, reduction='none', eps=1e-6):
        super().__init__()
        self.negative_ratio = negative_ratio
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, gt, mask):
        """
        :param pred: [N, 1, H, W]
        :param gt: [N, 1, H, W]
        :param mask: [N, H, W]
        :return:
        """
        positive = (gt * mask).byte()
        negative = ((1 - gt) * mask).byte()
        no_positive = int(positive.float().sum())
        no_negative_expect = int(no_positive * self.negative_ratio)
        no_negative_current = int(negative.float().sum())
        no_negative = min(no_negative_expect, no_negative_current)
        loss = F.binary_cross_entropy(pred, gt, reduction=self.reduction)
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()
        negative_loss, _ = torch.topk(negative_loss.view(-1), no_negative)
        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (no_negative + no_positive + self.eps)
        return balance_loss


class L1Loss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-6):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, gt, mask):
        if mask is not None:
            loss = (torch.abs(pred - gt) * mask).sum() / (mask.sum() + self.eps)
        else:
            loss = torch.nn.L1Loss(reduction=self.reduction)(pred, gt)
        return loss


class DiceLoss(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, pred, gt, mask):
        intersection = (pred * gt * mask).sum()
        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        loss = 1 - 2.0 * intersection / union
        assert loss <= 1
        return loss


class DBLoss(nn.Module):
    def __init__(self, alpha=1., beta=10., reduction='mean', negative_ratio=3, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        self.negative_ratio = negative_ratio
        self.eps = eps
        self.ohem_loss = OHEMBalanceCrossEntropyLoss(negative_ratio, reduction, eps)
        self.l1_loss = L1Loss(reduction, eps)
        self.dice_loss = DiceLoss(eps)

    def forward(self, preds, gts):
        """
        :param preds: probability map (Ls), binary map (Lb), threshold map (Lt)
        :param gts: prob map, binary map
        :return: prob_loss, threshold_loss, binary_loss, prob_threshold_loss, total_loss
        total_loss = Ls + alpha * Lb + beta * Lt
        """
        assert preds.dim() == 4
        assert gts.dim() == 4
        prob_map = preds[:, 0, :, :]
        threshold_map = preds[:, 1, :, :]
        if preds.size(1) == 3:
            appro_binary_map = preds[:, 2, :, :]
        prob_gt_map = gts[0, :, :, :]
        supervision_mask = gts[1, :, :, :]  # 0/1
        threshold_gt_map = gts[2, :, :, :]  # 0.3/0.7
        text_area_gt_map = gts[3, :, :, :]  # 0/1

        # loss
        prob_loss = self.ohem_loss(prob_map, prob_gt_map, supervision_mask)
        threshold_loss = self.l1_loss(threshold_map, threshold_gt_map, text_area_gt_map)
        prob_threshold_loss = prob_loss + self.beta * threshold_loss
        if preds.size(1) == 3:
            binary_loss = self.dice_loss(appro_binary_map, prob_gt_map, supervision_mask)
            total_loss = prob_threshold_loss + self.alpha * binary_loss
            return prob_loss, threshold_loss, binary_loss, prob_threshold_loss, total_loss
        else:
            return prob_threshold_loss
