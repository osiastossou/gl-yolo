"""
CCP-YOLO Custom Modules Implementation
Modules personnalisés pour l'architecture CCP-YOLO
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict
from pathlib import Path
import sys


class DilatedReparameterizationBlock(nn.Module):
    """
    Dilated Reparameterization Block (DRB)
    Convertit plusieurs branches parallèles en une convolution équivalente
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, dilation_rates=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        if dilation_rates is None:
            if kernel_size == 5:
                dilation_rates = [(1, 1), (1, 1), (2, 1)]  # (kernel, dilation)
            else:  # kernel_size == 7
                dilation_rates = [(1, 1), (1, 1), (2, 1), (3, 1)]
        
        self.dilation_rates = dilation_rates
        self.branches = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.paddings = []  # (left, right, top, bottom) per branch, for asymmetric "same" padding

        for k, d in dilation_rates:
            total_pad = d * (k - 1)
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            self.paddings.append((pad_before, pad_after, pad_before, pad_after))
            self.branches.append(
                nn.Conv2d(in_channels, out_channels, k, padding=0,
                         dilation=d, groups=in_channels, bias=False)
            )
            self.bn_layers.append(nn.BatchNorm2d(out_channels))

    def forward(self, x):
        outputs = []
        for branch, bn, pad in zip(self.branches, self.bn_layers, self.paddings):
            outputs.append(bn(branch(F.pad(x, pad))))
        return sum(outputs)


class MultiScaleDilatedModule(nn.Module):
    """
    C3k2_MSD: Multi-Scale Dilated Reparameterization Feature Extraction Module
    """
    def __init__(self, in_channels, out_channels, e=0.5):
        super().__init__()
        hidden = int(in_channels * e)
        
        # Réduction initiale
        self.conv1 = nn.Conv2d(in_channels, hidden, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden)
        
        # Trois branches parallèles
        # Branche 1: Conv 3x3 standard
        self.branch1 = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU()
        )
        
        # Branche 2: DRB avec kernel size 5
        self.branch2 = nn.Sequential(
            DilatedReparameterizationBlock(hidden, hidden, kernel_size=5),
            nn.BatchNorm2d(hidden),
            nn.ReLU()
        )
        
        # Branche 3: DRB avec kernel size 7
        self.branch3 = nn.Sequential(
            DilatedReparameterizationBlock(hidden, hidden, kernel_size=7),
            nn.BatchNorm2d(hidden),
            nn.ReLU()
        )
        
        # Fusion finale
        self.conv_fusion = nn.Conv2d(hidden * 3, out_channels, 1)
        self.bn_fusion = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        
        x = self.relu(self.bn1(self.conv1(x)))
        
        # Concatener les trois branches
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        
        x = torch.cat([b1, b2, b3], dim=1)
        x = self.relu(self.bn_fusion(self.conv_fusion(x)))
        
        # Connexion résiduelle adaptée
        if x.shape == residual.shape:
            x = x + residual
        
        return x


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation Attention Module"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class InteractiveAdaptiveFeatureFusion(nn.Module):
    """
    IAFM: Interactive Adaptive Feature Fusion Module
    Fusionne adaptativement deux branches de features
    """
    def __init__(self, c0, c1, reduction=16):
        super().__init__()
        
        # Fusion semantique
        self.semantic_agg = nn.Sequential(
            nn.Conv2d(c0 + c1, c0 + c1, 1),
            nn.BatchNorm2d(c0 + c1)
        )
        
        # SE Attention
        self.se = SqueezeExcitation(c0 + c1, reduction)
        
        # Adaptive weighting
        total_channels = c0 + c1
        self.adaptive_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(total_channels, total_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(total_channels // reduction, total_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x0, x1):
        # Adapter les dimensions si nécessaire
        if x0.shape[1] != x1.shape[1]:
            if x0.shape[1] > x1.shape[1]:
                x1 = F.interpolate(x1, size=x0.shape[2:], mode='bilinear', align_corners=False)
            else:
                x0 = F.interpolate(x0, size=x1.shape[2:], mode='bilinear', align_corners=False)
        
        # Agrégation sémantique
        x_cat = torch.cat([x0, x1], dim=1)
        x_cat = self.semantic_agg(x_cat)
        
        # SE attention
        x_cat = self.se(x_cat)
        
        # Poids adaptatifs
        weights = self.adaptive_weight(x_cat)
        
        # Scinder les poids
        c0, c1 = x0.shape[1], x1.shape[1]
        w0 = weights[:, :c0, :, :]
        w1 = weights[:, c0:, :, :]
        
        # Fusion interactive résiduelle
        x0_w = x0 * w0
        x1_w = x1 * w1
        
        y0 = x0 + x1_w
        y1 = x1 + x0_w
        
        # Concaténation finale
        output = torch.cat([y0, y1], dim=1)
        
        return output


class ProgressiveSharedWeightContextAggregation(nn.Module):
    """
    PSWCA: Progressive Shared-Weight Context Aggregation Module
    Remplace le SPPF pour une agrégation contextuelle progressive
    """
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        
        hidden = in_channels // 2
        
        # Réduction initiale
        self.reduce = nn.Conv2d(in_channels, hidden, 1)
        
        # Convolutions dilatées partagées
        self.shared_conv = nn.Conv2d(hidden, hidden, 3, padding=1, bias=False)
        self.bn_shared = nn.BatchNorm2d(hidden)
        
        # Expansion finale
        self.expand = nn.Conv2d(hidden * 4, out_channels, 1)
    
    def forward(self, x):
        x = self.reduce(x)
        
        # Progressive aggregation avec dilation rates [1, 3, 5]
        features = [x]
        
        # y1 avec dilation 1
        y1 = F.conv2d(x, self.shared_conv.weight, padding=1, dilation=1)
        y1 = self.bn_shared(y1)
        y1 = F.relu(y1)
        features.append(y1)
        
        # y2 avec dilation 3
        y2 = F.conv2d(y1, self.shared_conv.weight, padding=3, dilation=3)
        y2 = self.bn_shared(y2)
        y2 = F.relu(y2)
        features.append(y2)
        
        # y3 avec dilation 5
        y3 = F.conv2d(y2, self.shared_conv.weight, padding=5, dilation=5)
        y3 = self.bn_shared(y3)
        y3 = F.relu(y3)
        features.append(y3)
        
        # Concaténation et fusion
        x_out = torch.cat(features, dim=1)
        x_out = self.expand(x_out)
        
        return x_out


class WiseInnerMPDIoULoss(nn.Module):
    """
    Wise-Inner-MPDIoU Loss Function
    Combine Wise-IoU, Inner-IoU et MPDIoU
    """
    def __init__(self, alpha=1.7, delta=2.7, ratio=1.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.delta = delta
        self.ratio = ratio
        self.reduction = reduction
        self.eps = 1e-7
    
    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes: [N, 4] predicted boxes (x1, y1, x2, y2)
            target_boxes: [N, 4] target boxes (x1, y1, x2, y2)
        """
        
        # Calculer IoU standard
        iou = self._compute_iou(pred_boxes, target_boxes)
        
        # Wise-IoU component (quality-aware)
        loss_iou = 1 - iou
        wise_weight = self._compute_wise_weight(pred_boxes, target_boxes)
        
        # Inner-IoU component (auxiliary box)
        inner_iou = self._compute_inner_iou(pred_boxes, target_boxes, self.ratio)
        inner_loss = 1 - inner_iou
        
        # MPDIoU component (corner points)
        mpd_loss = self._compute_mpd_loss(pred_boxes, target_boxes)
        
        # Combinaison
        loss = wise_weight * (loss_iou + inner_loss + mpd_loss)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def _compute_iou(self, boxes1, boxes2):
        """Compute standard IoU"""
        inter = self._compute_inter(boxes1, boxes2)
        union = self._compute_union(boxes1, boxes2)
        iou = inter / (union + self.eps)
        return iou
    
    def _compute_inter(self, boxes1, boxes2):
        """Compute intersection area"""
        x1_min = torch.max(boxes1[:, 0], boxes2[:, 0])
        y1_min = torch.max(boxes1[:, 1], boxes2[:, 1])
        x2_max = torch.min(boxes1[:, 2], boxes2[:, 2])
        y2_max = torch.min(boxes1[:, 3], boxes2[:, 3])
        
        inter_w = (x2_max - x1_min).clamp(min=0)
        inter_h = (y2_max - y1_min).clamp(min=0)
        
        return inter_w * inter_h
    
    def _compute_union(self, boxes1, boxes2):
        """Compute union area"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        inter = self._compute_inter(boxes1, boxes2)
        return area1 + area2 - inter
    
    def _compute_wise_weight(self, boxes1, boxes2):
        """Compute Wise-IoU quality-aware weight"""
        iou = self._compute_iou(boxes1, boxes2)
        
        # Centroid distances
        cx1 = (boxes1[:, 0] + boxes1[:, 2]) / 2
        cy1 = (boxes1[:, 1] + boxes1[:, 3]) / 2
        cx2 = (boxes2[:, 0] + boxes2[:, 2]) / 2
        cy2 = (boxes2[:, 1] + boxes2[:, 3]) / 2
        
        dist = torch.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2 + self.eps)
        
        # Outlier degree
        beta = dist / (iou + self.eps)
        
        # Non-monotonic focusing mechanism
        r = beta / (self.delta * self.alpha * beta - self.delta + self.eps)
        
        return r.clamp(min=0, max=1)
    
    def _compute_inner_iou(self, boxes1, boxes2, ratio=1.0):
        """Compute Inner-IoU with auxiliary boxes"""
        # Auxiliary boxes
        w1 = boxes1[:, 2] - boxes1[:, 0]
        h1 = boxes1[:, 3] - boxes1[:, 1]
        
        w2 = boxes2[:, 2] - boxes2[:, 0]
        h2 = boxes2[:, 3] - boxes2[:, 1]
        
        # Boxes auxiliaires
        aux1_x1 = boxes1[:, 0] + (w1 * (1 - ratio)) / 2
        aux1_y1 = boxes1[:, 1] + (h1 * (1 - ratio)) / 2
        aux1_x2 = boxes1[:, 2] - (w1 * (1 - ratio)) / 2
        aux1_y2 = boxes1[:, 3] - (h1 * (1 - ratio)) / 2
        
        aux2_x1 = boxes2[:, 0] + (w2 * (1 - ratio)) / 2
        aux2_y1 = boxes2[:, 1] + (h2 * (1 - ratio)) / 2
        aux2_x2 = boxes2[:, 2] - (w2 * (1 - ratio)) / 2
        aux2_y2 = boxes2[:, 3] - (h2 * (1 - ratio)) / 2
        
        aux_boxes1 = torch.stack([aux1_x1, aux1_y1, aux1_x2, aux1_y2], dim=1)
        aux_boxes2 = torch.stack([aux2_x1, aux2_y1, aux2_x2, aux2_y2], dim=1)
        
        return self._compute_iou(aux_boxes1, aux_boxes2)
    
    def _compute_mpd_loss(self, boxes1, boxes2):
        """Compute corner-point distance loss"""
        # Top-left et bottom-right corners
        corners1_tl = boxes1[:, :2]
        corners1_br = boxes1[:, 2:]
        
        corners2_tl = boxes2[:, :2]
        corners2_br = boxes2[:, 2:]
        
        # Distances euclidiennes
        d_tl = torch.sqrt(((corners1_tl - corners2_tl) ** 2).sum(dim=1) + self.eps)
        d_br = torch.sqrt(((corners1_br - corners2_br) ** 2).sum(dim=1) + self.eps)
        
        # Normalisation
        max_dist = torch.sqrt(torch.tensor(2.0))  # Diagonal maximale normalisée
        
        mpd_loss = (d_tl + d_br) / (2 * max_dist + self.eps)
        
        return mpd_loss
 
 
class C3k2_MSD(nn.Module):
    """
    C3k2 avec Multi-Scale Dilated Module
    Remplace le C3k2 standard dans le YAML
    """
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        c_ = int(c2 * e)
        
        from ultralytics.nn.modules import Conv
        
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        
        # Modules MSD empilés
        self.m = nn.Sequential(
            *[MultiScaleDilatedModule(c_, c_) for _ in range(n)]
        )
    
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
 
 
class PSWCA(nn.Module):
    """
    Progressive Shared-Weight Context Aggregation Module
    Remplace SPPF pour meilleure préservation des détails spatiaux
    """
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        
        from ultralytics.nn.modules import Conv
        
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1)
        self.pswca = ProgressiveSharedWeightContextAggregation(c_, c_)
    
    def forward(self, x):
        x = self.cv1(x)
        y0 = x
        y1 = self.pswca(x)
        y2 = self.pswca(y1)
        y3 = self.pswca(y2)
        return self.cv2(torch.cat((y0, y1, y2, y3), 1))
 
 
class IAFM(nn.Module):
    """
    Interactive Adaptive Feature Fusion Module
    Pour la fusion adaptative des features
    """
    def __init__(self, c1, c2):
        super().__init__()
        from ultralytics.nn.modules import Conv

        # c1: channels of the primary (-1) branch, c2: channels of the skip branch.
        # InteractiveAdaptiveFeatureFusion requires both inputs to share the same
        # channel count, so the skip branch is projected onto c1 first.
        self.align = Conv(c2, c1, 1, 1) if c1 != c2 else None
        self.fusion = InteractiveAdaptiveFeatureFusion(c1, c1, reduction=16)
        self.proj = Conv(c1 * 2, c1, 1, 1)

    def forward(self, x):
        x1, x2 = x
        if self.align is not None:
            x2 = self.align(x2)
        x = self.fusion(x1, x2)
        return self.proj(x)
