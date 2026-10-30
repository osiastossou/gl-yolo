"""Demonstration 2: the pooling cascade is an information bottleneck; the
CSP shortcut branch is not.

Two exact, gradient-based checks (no training involved):
  (a) A max-pool cascade -- the only aggregation mechanism plain SPPF uses --
      has exactly zero gradient at any input position that is not the
      arg-max of the window(s) it participates in. That position's value
      cannot influence the output at all.
  (b) A plain Conv1x1 branch (the CSP shortcut used by SPPFCSPC-CBAM) has
      nonzero gradient at every input position: it discards nothing.

Usage:
    python demo2_pooling_bottleneck.py
"""

import torch
import torch.nn as nn

torch.manual_seed(0)


def demo_pooling_cascade(C=8, H=6, W=6, k=5, n=3) -> None:
    x = torch.randn(1, C, H, W, requires_grad=True)
    pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    y = x
    for _ in range(n):  # exactly SPPF's pooling cascade (self.m applied n times)
        y = pool(y)
    y.sum().backward()

    grad = x.grad[0]
    window_max = pool(x.detach())[0]
    is_argmax = x.detach()[0] >= window_max - 1e-6

    print("=== (a) Max-pool cascade gradient (SPPF's only aggregation mechanism) ===")
    print(f"positions that are NOT a window argmax : {(~is_argmax).sum().item()} / {x.numel()}")
    print(f"max |grad| at non-argmax positions       : {grad[~is_argmax].abs().max().item():.10f}  (exactly 0)")
    print(f"max |grad| at argmax positions            : {grad[is_argmax].abs().max().item():.6f}  (nonzero)")


def demo_csp_shortcut(C=8, H=6, W=6) -> None:
    # Plain Conv1x1, no BatchNorm: BN with batch=1 in train mode computes
    # degenerate single-sample statistics that distort the gradient scale --
    # irrelevant to the point being demonstrated (the shortcut *path*, not BN).
    cv2 = nn.Conv2d(C, C, 1, bias=False)
    x2 = torch.randn(1, C, H, W, requires_grad=True)
    cv2(x2).sum().backward()
    g = x2.grad[0].abs()

    print("\n=== (b) CSP shortcut branch gradient (SPPFCSPC-CBAM's parallel path) ===")
    print(f"positions with exactly-zero gradient : {(g < 1e-12).sum().item()} / {g.numel()}")
    print(f"min / max |grad| over all positions   : {g.min().item():.6f} / {g.max().item():.6f}  (nonzero everywhere)")


if __name__ == "__main__":
    demo_pooling_cascade()
    demo_csp_shortcut()
