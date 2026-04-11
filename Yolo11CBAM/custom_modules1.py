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


import torch.nn.functional as F
from typing import List

from ultralytics.nn.modules.conv import Conv



# --- REPLACE YOUR WTConv CLASS WITH THIS ONE ---
class WaveletAttentionBlock(nn.Module):

    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size

        # Filtres Haar (Low-pass et High-pass)
        lp = torch.tensor([1.0, 1.0]) / 2**0.5
        hp = torch.tensor([1.0, -1.0]) / 2**0.5

        # Création des filtres 2D pour conv2d
        self.register_buffer('ll', torch.ger(lp, lp).unsqueeze(0).unsqueeze(0))
        self.register_buffer('lh', torch.ger(lp, hp).unsqueeze(0).unsqueeze(0))
        self.register_buffer('hl', torch.ger(hp, lp).unsqueeze(0).unsqueeze(0))
        self.register_buffer('hh', torch.ger(hp, hp).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, C, H, W = x.shape

        # Appliquer sur chaque canal
        bands = []
        for filt in [self.ll, self.lh, self.hl, self.hh]:
            # Répliquer le filtre pour chaque canal
            f = filt.repeat(C, 1, 1, 1)  # [C,1,2,2]
            band = F.conv2d(x, f, stride=2, groups=C)  # downsample par 2
            bands.append(band)

        # Concaténer les 4 sous-bandes : shape = [B, 4C, H/2, W/2]
        out = torch.cat(bands, dim=1)

        # Découper en patchs pour attention : [B, num_patches, patch_dim]
        patches = out.unfold(2, self.patch_size, self.patch_size) \
                     .unfold(3, self.patch_size, self.patch_size)  # [B, 4C, nH, nW, pH, pW]
        patches = patches.contiguous().view(B, out.shape[1], -1, self.patch_size*self.patch_size)  # [B, 4C, num_patches, pH*pW]
        patches = patches.permute(0, 2, 1, 3).contiguous().view(B, -1, self.patch_size*self.patch_size*out.shape[1])  # [B, num_patches, patch_dim]

        return patches  # prêt pour multi-head attention
    


class GL_CAB(nn.Module):
    """
    Implementation of the Global-Local Combined Attention Block (GL-CAB)
    from the paper "Global-Local Attention Mechanism Based Small Object Detection".
    
    This module creates an attention map by combining global context, local features,
    and local detail features to emphasize small objects that might be lost.
    """
    def __init__(self, c1, c2=None):
        # c1: input channels
        # c2: output channels (defaults to c1 if not provided)
        super(GL_CAB, self).__init__()
        if c2 is None:
            c2 = c1
            
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
            nn.Hardswish()
        )

        # --- Path for Local Detail Features: L(P) in the paper ---
        # Refines details using global context as a guide.
        # This path takes the output of the global average pooling.
        self.local_detail_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Global Average Pooling
            conv_block(c1, c1),
            nn.Hardswish()
        )
        
        self.sigmoid = nn.Sigmoid()
        
        # Final convolution to ensure the output channel count is correct
        self.final_conv = nn.Conv2d(c1, c2, kernel_size=1) if c1 != c2 else nn.Identity()

    def forward(self, x):
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
    



class GL_CAB_Light(nn.Module):
    def __init__(self, c1, c2=None, reduction=4):
        super().__init__()
        if c2 is None:
            c2 = c1
        rc = max(8, c1 // reduction)
        self.reduce = nn.Conv2d(c1, rc, 1)
        self._global = nn.Sequential(
            nn.Conv2d(rc, rc, 1, bias=False),
            nn.BatchNorm2d(rc),
            nn.SiLU()
        )
        self.local = nn.Sequential(
            nn.Conv2d(rc, rc, 1, bias=False),
            nn.BatchNorm2d(rc),
            nn.SiLU()
        )
        self.detail = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(rc, rc, 1, bias=False),
            nn.BatchNorm2d(rc),
            nn.SiLU()
        )
        self.fuse = nn.Conv2d(rc, rc, 1)
        self.sigmoid = nn.Sigmoid()
        self.expand = nn.Conv2d(rc, c2, 1)

    def forward(self, x):
        x_r = self.reduce(x)
        g = self._global(x_r)
        l = self.local(x_r)
        d = self.detail(x_r)
        attn = self.sigmoid(self.fuse(l + d))
        out = g * attn
        return self.expand(out)