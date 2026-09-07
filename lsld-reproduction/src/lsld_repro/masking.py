from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import torch
from torch import nn
MaskMode = Literal['all_on', 'stochastic', 'deterministic', 'external']

def gumbel_sigmoid_straight_through(logits: torch.Tensor, temperature: float=1.0, eps: float=1e-10) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    uniforms = torch.rand((2, *logits.shape), device=logits.device, dtype=logits.dtype)
    noise = -torch.log(torch.log(uniforms[1] + eps) / torch.log(uniforms[0] + eps) + eps)
    soft = torch.sigmoid((logits + noise) / temperature)
    hard = (soft > 0.5).to(soft.dtype)
    return (hard - soft).detach() + soft

@dataclass(frozen=True)
class HeadLocation:
    layer: int
    head: int

class AttentionHeadMask(nn.Module):

    def __init__(self, model, *, temperature: float=1.0) -> None:
        super().__init__()
        backbone = getattr(model, 'model', None)
        layers = getattr(backbone, 'layers', None)
        if layers is None:
            raise TypeError('Expected a Llama-style model with `.model.layers[*].self_attn.o_proj`.')
        self.n_layers = len(layers)
        self.n_heads = int(model.config.num_attention_heads)
        self.head_dim = int(getattr(model.config, 'head_dim', int(model.config.hidden_size) // self.n_heads))
        self.temperature = float(temperature)
        device = next(model.parameters()).device
        self.mask_logits = nn.Parameter(torch.empty(self.n_layers, self.n_heads, device=device, dtype=torch.float32))
        nn.init.normal_(self.mask_logits, mean=0.0, std=0.01)
        self.mode: MaskMode = 'all_on'
        self.register_buffer('external_mask', torch.ones(self.n_layers, self.n_heads, device=device, dtype=torch.float32))
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        for layer_idx, layer in enumerate(layers):
            attention = getattr(layer, 'self_attn', None)
            output_projection = getattr(attention, 'o_proj', None)
            if output_projection is None:
                self.close()
                raise TypeError(f'Layer {layer_idx} has no `self_attn.o_proj`; unsupported architecture.')
            self._handles.append(output_projection.register_forward_pre_hook(self._make_hook(layer_idx)))

    @property
    def total_heads(self) -> int:
        return self.n_layers * self.n_heads

    def _mask_for_layer(self, layer_idx: int) -> torch.Tensor:
        if self.mode == 'all_on':
            return torch.ones_like(self.mask_logits[layer_idx])
        if self.mode == 'stochastic':
            return gumbel_sigmoid_straight_through(self.mask_logits[layer_idx], self.temperature)
        if self.mode == 'deterministic':
            return (self.mask_logits[layer_idx] > 0).to(self.mask_logits.dtype)
        if self.mode == 'external':
            return self.external_mask[layer_idx]
        raise RuntimeError(f'Unknown mask mode: {self.mode}')

    def _make_hook(self, layer_idx: int):

        def hook(_module, args):
            if not args:
                raise RuntimeError('Attention output projection received no positional input')
            hidden = args[0]
            expected = self.n_heads * self.head_dim
            if hidden.shape[-1] != expected:
                raise RuntimeError(f'o_proj input width {hidden.shape[-1]} does not equal num_heads*head_dim ({expected})')
            original_shape = hidden.shape
            split = hidden.reshape(*original_shape[:-1], self.n_heads, self.head_dim)
            mask = self._mask_for_layer(layer_idx).to(dtype=hidden.dtype)
            masked = split * mask.view(*[1] * (split.ndim - 2), self.n_heads, 1)
            return (masked.reshape(original_shape), *args[1:])
        return hook

    def keep_probabilities(self) -> torch.Tensor:
        return torch.sigmoid(self.mask_logits)

    def deterministic_mask(self) -> torch.Tensor:
        return (self.mask_logits > 0).to(torch.float32)

    def set_external_mask(self, mask: torch.Tensor) -> None:
        if tuple(mask.shape) != (self.n_layers, self.n_heads):
            raise ValueError(f'Mask shape {tuple(mask.shape)} does not match {(self.n_layers, self.n_heads)}')
        self.external_mask.copy_(mask.to(self.external_mask))
        self.mode = 'external'

    def removed_heads(self, mask: torch.Tensor | None=None) -> list[HeadLocation]:
        mask = self.deterministic_mask() if mask is None else mask
        return [HeadLocation(layer=layer, head=head) for layer in range(self.n_layers) for head in range(self.n_heads) if float(mask[layer, head]) == 0.0]

    def close(self) -> None:
        while self._handles:
            self._handles.pop().remove()

    def __del__(self):
        self.close()
