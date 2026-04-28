# custom_modules.py

import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    """
    Module d'attention sur les canaux (Channel Attention).
    Il apprend à donner plus de poids aux canaux de caractéristiques les plus pertinents.
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """
    Module d'attention spatiale (Spatial Attention).
    Il apprend à se concentrer sur les régions spatiales les plus importantes.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x_cat)
        return self.sigmoid(x)

class CBAM(nn.Module):
    """
    Module CBAM complet que nous insérerons dans YOLOv8.
    Il applique séquentiellement l'attention canal puis l'attention spatiale.
    """
    def __init__(self, c1, ratio=16, kernel_size=7): # c1 est le nombre de canaux d'entrée
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(c1, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class TransformerBlock(nn.Module):
    """
    Bloc Transformer pour capturer les dépendances globales, inspiré du module C3.
    """
    # MODIFIÉ : La signature accepte c2, num_heads, et num_layers
    def __init__(self, c1, c2, num_heads=4, num_layers=1):
        super().__init__()
        assert c1 == c2, "Les canaux d'entrée et de sortie doivent être identiques pour le TransformerBlock"
        
        # Le Transformer Encoder de PyTorch
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c1, 
            nhead=num_heads, 
            dim_feedforward=c1 * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # Le reste de la fonction forward est inchangé
        b, c, h, w = x.shape
        pos_embed = x.flatten(2).permute(0, 2, 1)
        transformed = self.transformer_encoder(pos_embed)
        output = transformed.permute(0, 2, 1).view(b, c, h, w)
        return output

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, 
                               kernel_size=patch_size, 
                               stride=patch_size)  # "Découpage intelligent"

    def forward(self, x):
        x = self.proj(x)  # [B, D, H/P, W/P]
        x = x.flatten(2)  # [B, D, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, D]
        return x
    
###############@@@

# ultralytics/nn/modules/wavelet_backbone.py




# gl_cab.py


import torch
import torch.nn as nn
import torch.nn.functional as F

class GL_CAB1(nn.Module):
    def __init__(self, channels, kernel_size=3, fusion='add'):
        """
        channels: number of input channels (C)
        kernel_size: kernel for local conv (default 3)
        fusion: 'add' or 'cat'
        """
        super().__init__()
        assert fusion in ('add', 'cat')

        padding = kernel_size // 2

        # Local branch
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(channels),
            nn.Hardswish(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Global branch
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),               # (B, C, 1, 1)
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(channels),
        )

        self.fusion = fusion
        if fusion == 'cat':
            self.fuse_conv = nn.Sequential(
                nn.Conv2d(2*channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.Hardswish(inplace=True),
            )
        else:
            self.fuse_conv = nn.Identity()

        # Output mapping W(P)
        self.output_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        # x: [B, C, H, W]
        local = self.local(x)                       # [B, C, H, W]
        glob = self.global_branch(x)                # [B, C, 1, 1]
        glob = glob.expand_as(local)                # broadcast -> [B, C, H, W]

        if self.fusion == 'cat':
            fused = torch.cat([local, glob], dim=1) # [B, 2C, H, W]
            fused = self.fuse_conv(fused)          # -> [B, C, H, W]
        else:
            fused = local + glob                   # element-wise add

        att = torch.sigmoid(self.output_conv(fused))# attention map in (0,1)
        out = att * fused                           # re-weighted features
        return out

from typing import List

from ultralytics.nn.modules.block import PSABlock
from ultralytics.nn.modules.block import ABlock
class GL_CAB(nn.Module):

    """
    Implementation of the Global-Local Combined Attention Block (GL-CAB)
    from the paper "Global-Local Attention Mechanism Based Small Object Detection".
    
    This module creates an attention map by combining global context, local features,
    and local detail features to emphasize small objects that might be lost.
    """
    def __init__(self, c1, c2=None, n=2):
        # c1: input channels
        # c2: output channels (defaults to c1 if not provided)
        super(GL_CAB, self).__init__()
        if c2 is None:
            c2 = c1

        self.c1 = c1
        self.c2 = c2
        self.n = n
            
        # A reusable 1x1 convolution block with BatchNorm
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # --- Path for Global Features: G(P) in the paper ---
        # Captures the overall scene context.
        self.global_features_path = nn.Sequential(
            conv_block(c1, c1),
            nn.ReLU()
        )

        # --- Path for Local Features: Z(P) in the paper ---
        # Focuses on local information without global context.
        self.local_features_path = nn.Sequential(
            conv_block(c1, c1),
            nn.SiLU()
        )

        # --- Path for Local Detail Features: L(P) in the paper ---
        # Refines details using global context as a guide.
        # This path takes the output of the global average pooling.
        self.local_detail_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Global Average Pooling
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.SiLU()
        )
        
        self.sigmoid = nn.Sigmoid()
        
        # Final convolution to ensure the output channel count is correct
        self.final_conv = nn.Conv2d(c1, c2, kernel_size=1) if c1 != c2 else nn.Identity()

    def forward(self, x,n=2):
        """
        Forward pass of the GL-CAB module.
        Follows Formula (5): W(P) = σ(L(P) ⊕ Z(P)) ⊗ G(P)
        The attention map is applied to the globally-processed features.
        """
        # P is the input tensor x
        
        # G(P): Processed global features
        global_features_processed = self.global_features_path(x)
        
        # L(P): Local details derived from global context
        local_detail_features = self.local_detail_path(x)
        
        # Z(P): Processed local features
        local_features_processed = self.local_features_path(x)
        
        # L(P) ⊕ Z(P): Fusion of local and detail features (element-wise addition)
        fused_local_info = local_detail_features + local_features_processed
        
        # σ(...): The sigmoid function creates the final attention map
        attention_map = self.sigmoid(fused_local_info)
        
        # ⊗ G(P): The attention map is applied to the processed global features
        out = attention_map * global_features_processed
        
        return self.final_conv(out)
    

class GL_CAB_PSABlock(nn.Module):

    """
    Implementation of the Global-Local Combined Attention Block (GL-CAB)
    from the paper "Global-Local Attention Mechanism Based Small Object Detection".

    This module creates an attention map by combining global context, local features,
    and local detail features to emphasize small objects that might be lost.
    """
    def __init__(
        self,
        c1,
        c2=None,
        n=2,
        a2=True,
        area=1,
        residual=False,
        mlp_ratio=2.0,
        e=0.5,
        g=1,
        shortcut=True,
    ):
        # c1: input channels
        # c2: output channels (defaults to c1 if not provided)
        super(GL_CAB_PSABlock, self).__init__()
        if c2 is None:
            c2 = c1

        self.c1 = c1
        self.c2 = c2
        self.n = n

        num_heads = max(1, self.c1 // 64)
        if a2:
            self.m = nn.Sequential(*(PSABlock(self.c1, num_heads=num_heads, mlp_ratio=mlp_ratio, area=area) for _ in range(n)))
        else:
            self.m = nn.Sequential(*(nn.Identity() for _ in range(n)))

        # A reusable 1x1 convolution block with BatchNorm
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # --- Path for Global Features: G(P) in the paper ---
        # Captures the overall scene context.
        self.global_features_path = nn.Sequential(
            conv_block(c1, c1),
            nn.ReLU()
        )

        # --- Path for Local Features: Z(P) in the paper ---
        # Focuses on local information without global context.
        self.local_features_path = nn.Sequential(
            conv_block(c1, c1),
            nn.SiLU()
        )

        # --- Path for Local Detail Features: L(P) in the paper ---
        # Refines details using global context as a guide.
        # This path takes the output of the global average pooling.
        self.local_detail_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Global Average Pooling
            nn.Conv2d(c1, c1, kernel_size=1, bias=False),
            nn.SiLU()
        )

        self.sigmoid = nn.Sigmoid()

        # Final convolution to ensure the output channel count is correct
        self.final_conv = nn.Conv2d(c1, c2, kernel_size=1) if c1 != c2 else nn.Identity()

    def forward(self, x,n=2):
        """
        Forward pass of the GL-CAB module.
        Follows Formula (5): W(P) = σ(L(P) ⊕ Z(P)) ⊗ G(P)
        The attention map is applied to the globally-processed features.
        """
        # P is the input tensor x

        # G(P): Processed global features
        global_features_processed = self.global_features_path(x)

        global_features_processed = self.m(global_features_processed)

        # L(P): Local details derived from global context
        local_detail_features = self.local_detail_path(x)

        # Z(P): Processed local features
        local_features_processed = self.local_features_path(x)

        # L(P) ⊕ Z(P): Fusion of local and detail features (element-wise addition)
        fused_local_info = local_detail_features + local_features_processed

        # σ(...): The sigmoid function creates the final attention map
        attention_map = self.sigmoid(fused_local_info)

        # ⊗ G(P): The attention map is applied to the processed global features
        out = attention_map * global_features_processed

        return self.final_conv(out)