
from typing import Dict, Union

import torch
import torch.nn as nn
from jaxtyping import Float

from transformer_lens.hook_points import HookPoint
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig


class RMSNormPre_identity(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):

        super().__init__()
        self.cfg = HookedTransformerConfig.unwrap(cfg)
        self.eps = self.cfg.eps

        # Adds a hook point for the normalisation scale factor
        self.hook_scale = HookPoint()  # [batch, pos]
        self.hook_normalized = HookPoint()  # [batch, pos, length]

    def forward(
        self, x: Float[torch.Tensor, "batch pos length"]
    ) -> Float[torch.Tensor, "batch pos length"]:
        if self.cfg.dtype not in [torch.float32, torch.float64]:
            x = x.to(torch.float32)

        scale: Float[torch.Tensor, "batch pos 1"] = self.hook_scale(
            (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        )
        return self.hook_normalized(x).to(self.cfg.dtype)  # [batch, pos, length]
