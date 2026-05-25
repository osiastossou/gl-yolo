import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableLPPool2d(nn.Module):
    """Version apprenable du LPPool2d. Le paramètre 'p' est optimisé par rétropropagation pendant l'entraînement.
    """

    def __init__(self, kernel_size, stride=None, initial_p=2.0, ceil_mode=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.ceil_mode = ceil_mode

        # On définit 'p' comme un paramètre apprenable
        # Initialisé par défaut à 2.0 (norme euclidienne)
        self.p = nn.Parameter(torch.tensor([float(initial_p)]))

    def forward(self, input):
        # Sécurité : p doit être >= 1 pour que la norme Lp soit définie
        # On utilize F.softplus ou clamp pour garantir que p ne devienne pas < 1
        p_clamped = torch.clamp(self.p, min=1.0)

        # On utilize la function fonctionnelle de PyTorch
        return F.lp_pool2d(input, float(p_clamped), self.kernel_size, self.stride, self.ceil_mode)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.item():.4f}, kernel_size={self.kernel_size}, stride={self.stride})"
