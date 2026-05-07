import torch
import torch.nn as nn
import torch.nn.functional as F


class PWC_Layer(nn.Module):
    """
    Convolution Pondérée par Probabilités (PWC) - Version 1.0 (2026).
    Amplifie les pixels rares pour la détection de petits objets. [cite: 4, 8]
    """

    def __init__(self, c1, c2, k=3, s=1, p=1, bins=17):
        super().__init__()
        self.k = k  # Taille du noyau (ex: 3) [cite: 67]
        self.s = s
        self.p = p
        self.bins = bins  # Histogramme à 17 bins
        self.eps = 1e-6

        # Filtre de convolution standard appris W [cite: 104]
        self.weight = nn.Parameter(torch.randn(c2, c1, k, k))

    def forward(self, x):
        b, c, h, w = x.shape

        # 1. Extraction du patch local P_{i,j} [cite: 67]
        # On utilise unfold pour obtenir tous les patches du voisinage local
        patches = F.unfold(x, kernel_size=self.k, padding=self.p, stride=self.s)
        # patches shape: (b, C*k*k, L) avec L = nombre de positions spatiales

        N = patches.shape[1]  # Nombre d'éléments scalaires N = C*k*k [cite: 69]
        L = patches.shape[-1]

        # 2. Calcul des poids (Stop-Gradient / no_grad)
        # On traite les poids comme des constantes pour la rétropropagation
        with torch.no_grad():
            # Détermination du min et max du patch pour les bins adaptatifs [cite: 72, 78]
            p_min = patches.min(dim=1, keepdim=True)[0]
            p_max = patches.max(dim=1, keepdim=True)[0]

            # Calcul des indices de bin b(p_n) [cite: 74, 75]
            # On normalise la valeur du pixel dans l'intervalle [0, bins-1]
            bin_idx = ((patches - p_min) * self.bins / (p_max - p_min + self.eps)).long()
            bin_idx = torch.clamp(bin_idx, 0, self.bins - 1)

            # Calcul de la probabilité empirique p_hat [cite: 81, 82]
            # Initialisation avec le bon dtype/device pour éviter l'erreur scatter_add
            count = torch.zeros((b, self.bins, L), dtype=patches.dtype, device=patches.device)
            ones = torch.ones_like(patches)

            # On remplit l'histogramme pour chaque patch
            count.scatter_add_(1, bin_idx, ones)
            p_hat = count / N  # Fréquence relative de chaque bin [cite: 82]

            # Récupération de la probabilité associée à chaque pixel du patch
            p_pixel = torch.gather(p_hat, 1, bin_idx)

            # Calcul du poids inverse-probabilité w(p_n) [cite: 88, 89]
            w_raw = 1.0 / (p_pixel + self.eps)

            # Normalisation des poids (moyenne = 1 dans le patch) [cite: 92, 93]
            w_tilde = w_raw / (w_raw.mean(dim=1, keepdim=True) + self.eps)

        # 3. Opération PWC [cite: 101, 102]
        # Application de la pondération sur le patch original
        # Les pixels rares (objet) sont amplifiés, le fond est atténué [cite: 13, 96]
        weighted_patches = w_tilde * patches

        # Convolution finale : y = <W, P_tilde> [cite: 102]
        # On utilise matmul pour simuler la convolution sur les patches dépliés
        w_flat = self.weight.view(self.weight.shape[0], -1)
        y = torch.matmul(w_flat, weighted_patches)

        # 4. Redimensionnement vers la forme spatiale de sortie
        h_out = (h + 2 * self.p - self.k) // self.s + 1
        w_out = (w + 2 * self.p - self.k) // self.s + 1
        return y.view(b, -1, h_out, w_out)


class PWCConv(nn.Module):
    """
    Bloc conteneur incluant BatchNorm et Activation SiLU
    pour une intégration fluide dans le backbone YOLO.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = PWC_Layer(c1, c2, k, s, p)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))