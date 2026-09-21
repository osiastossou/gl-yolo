import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNAct(nn.Module):
    """Bloc utilitaire : Conv2d + BatchNorm2d + Activation"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, act_layer=nn.SiLU):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class HS_Conv_BN(nn.Module):
    """Sous-bloc composé de : Hardswish (HS) -> Conv 1x1 -> BatchNorm"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        # Functional (non in-place) Hardswish: Ultralytics' initialize_weights forces
        # inplace=True on nn.Hardswish modules, which corrupts tensors reused elsewhere
        # (e.g. `blue` and `green` in GL_PSWCA are also consumed by other branches).
        return self.bn(self.conv(F.hardswish(x, inplace=False)))


class GL_PSWCA(nn.Module):
    """Parse-model compatible progressive shared-weight context block.

    Ultralytics calls custom modules with the signature ``(c1, c2, *args, **kwargs)``;
    this class previously only accepted a single ``channels`` argument, which caused the
    YAML loader to pass extra positional arguments and crash during model build.
    """

    def __init__(self, c1: int, c2: int = None, k: int = 5, *args, **kwargs):
        super().__init__()

        # Keep compatibility with both legacy direct use and Ultralytics YAML parsing.
        c2 = c1 if c2 is None else c2
        c_mid = c2 // 2

        # 1. Conv 1x1 d'entrée : C -> c/2
        self.in_proj = nn.Conv2d(c1, c_mid, kernel_size=1, stride=1, padding=0, bias=False)

        # 2. Bloc vertical gauche : BN -> ReLU -> Conv 1x1 -> BN
        self.left_block = nn.Sequential(
            nn.BatchNorm2d(c_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_mid, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c_mid)
        )

        # 3. D-3 Conv 3x3 (Dilatation = 3, Padding = 3 pour conserver la taille H x W)
        self.d3_conv = nn.Conv2d(c_mid, c_mid, kernel_size=3, stride=1, padding=3, dilation=3, bias=False)

        # 4. Bloc HS -> Conv 1x1 -> BN sur la branche bleue
        self.branch_blue = HS_Conv_BN(c_mid)

        # 5. Bloc HS -> Conv 1x1 -> BN sur la branche verte (donne la feature orange)
        self.branch_green = HS_Conv_BN(c_mid)

        # 6. Projection finale : Concat (2C) -> C
        self.out_proj = nn.Conv2d(c_mid * 4, c2, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # Entrée : (B, C, H, W)

        # Tenseur bleu (c/2)
        blue = self.in_proj(x)

        # Tenseur jaune (c/2)
        yellow = self.left_block(blue)

        # Tenseur vert (c/2)
        green = self.d3_conv(yellow)

        # Branche bleue transformée (c/2)
        blue_trans = self.branch_blue(blue)

        # Tenseur orange (c/2)
        orange = self.branch_green(green)

        # Concaténation des 4 features (c/2 * 4 = 2c)
        concat_feat = torch.cat([blue_trans, yellow, green, orange], dim=1)

        # Projection de sortie vers C canaux
        out = self.out_proj(concat_feat)

        return out