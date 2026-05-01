
import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import ABlock

class AdaptA2C2f(nn.Module):
    """
    A2C2f avec area adaptatif appris.
    Le modèle apprend automatiquement :
    - area petit (4-8) sur P3 : petits objets denses
    - area grand (1-2) sur P5 : objets larges, contexte global
    """
    def __init__(self, c1, c2, n=1, a2=True, area=1,
                 residual=False, mlp_ratio=2.0, e=0.5,
                 g=1, shortcut=True,
                 area_candidates=(1, 2, 4)):
        super().__init__()
        c_ = int(c2 * e)
        assert c_ % 32 == 0

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)
        self.gamma = nn.Parameter(
            0.01 * torch.ones(c2), requires_grad=True
        ) if a2 and residual else None

        # Gate léger : apprend l'importance de chaque area
        n_areas = len(area_candidates)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c_, n_areas, bias=False),
            nn.Softmax(dim=-1),
        )

        # Un ABlock par area candidate
        self.branches = nn.ModuleList([
            nn.Sequential(*(
                ABlock(c_, c_ // 32, mlp_ratio, a)
                for _ in range(2)
            ))
            for a in area_candidates
        ])
        self.n_areas = n_areas

    def forward(self, x):
        y = [self.cv1(x)]

        feat = y[-1]
        B = feat.shape[0]

        # Poids appris pour chaque area
        weights = self.gate(feat)              # (B, n_areas)

        # Toutes les branches en parallèle
        outs = torch.stack(
            [branch(feat) for branch in self.branches],
            dim=1
        )                                      # (B, n_areas, C', H, W)

        # Fusion pondérée
        w = weights.view(B, self.n_areas, 1, 1, 1)
        fused = (outs * w).sum(dim=1)         # (B, C', H, W)

        y.append(fused)
        result = self.cv2(torch.cat(y, 1))

        if self.gamma is not None:
            return x + self.gamma.view(-1, self.gamma.shape[0], 1, 1) * result
        return result