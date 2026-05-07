import torch
import torch.nn as nn
import torch.nn.functional as F

class PWCConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1, bins=17):
        super().__init__()
        self.k = k
        self.s = s
        self.p = p
        self.bins = bins
        self.eps = 1e-6
        # Le filtre W est un filtre convolutif standard appris
        self.weight = nn.Parameter(torch.randn(c2, c1, k, k))

    def forward(self, x):
        # 1. Extraction des patches (unfold) [cite: 525]
        b, c, h, w = x.shape
        patches = F.unfold(x, kernel_size=self.k, padding=self.p, stride=self.s)
        # patches shape: (b, c*k*k, L) où L est le nombre de positions
        
        N = c * self.k * self.k
        L = patches.shape[-1]
        
        # 2. Calcul des poids (en mode no_grad / stop-gradient) [cite: 506]
        with torch.no_grad():
            p_min = patches.min(dim=1, keepdim=True)[0]
            p_max = patches.max(dim=1, keepdim=True)[0]
            
            # Calcul des indices de bins [cite: 378]
            bin_idx = ((patches - p_min) * self.bins / (p_max - p_min + self.eps)).long()
            bin_idx = torch.clamp(bin_idx, 0, self.bins - 1)
            
            # Probabilités empiriques (via scatter_add) [cite: 385, 528]
            # On compte l'occurrence de chaque bin par patch
            count = torch.zeros((b, self.bins, L), device=x.device)
            ones = torch.ones_like(patches)
            count.scatter_add_(1, bin_idx, ones)
            p_hat = count / N  # Probabilité empirique [cite: 385]
            
            # Récupération de la probabilité correspondant à chaque pixel
            p_pixel = torch.gather(p_hat, 1, bin_idx)
            
            # Calcul des poids normalisés [cite: 391, 396]
            w_raw = 1.0 / (p_pixel + self.eps)
            w_tilde = w_raw / (w_raw.mean(dim=1, keepdim=True) + self.eps)

        # 3. Application de la pondération et convolution [cite: 405]
        weighted_patches = w_tilde * patches
        
        # Convolution via matmul (plus rapide pour les patches dépliés) [cite: 530]
        w_flat = self.weight.view(self.weight.shape[0], -1)
        y = torch.matmul(w_flat, weighted_patches) # (b, c2, L)
        
        # Repliage vers la forme image
        h_out = (h + 2*self.p - self.k) // self.s + 1
        w_out = (w + 2*self.p - self.k) // self.s + 1
        return y.view(b, -1, h_out, w_out)