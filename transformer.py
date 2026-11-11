"""Transformer (encoder-decoder) implemented from scratch with PyTorch.

This module follows the original architecture from "Attention Is All You Need"
(Vaswani et al., 2017): multi-head scaled dot-product attention, position-wise
feed-forward networks, fixed sinusoidal positional encodings, and encoder /
decoder stacks that use residual connections followed by LayerNorm
(post-LN, the ordering used in the paper).

Only ``torch`` and the Python standard library are used. No training loop,
decoding loop, tokenizer, or data loader is included.

Usage example::

    import torch
    from transformer import Transformer

    model = Transformer(
        src_vocab_size=50,
        tgt_vocab_size=60,
        d_model=32,
        num_heads=4,
        d_ff=64,
        num_layers=2,
        dropout=0.1,
        pad_idx=0,
    )
    model.eval()  # disables dropout -> deterministic outputs

    src = torch.randint(1, 50, (2, 7))  # (batch, src_len) token ids
    tgt = torch.randint(1, 60, (2, 5))  # (batch, tgt_len) token ids
    with torch.no_grad():
        logits = model(src, tgt)        # (batch, tgt_len, tgt_vocab_size)
    print(logits.shape)                 # torch.Size([2, 5, 60])

Mask convention: masks are boolean tensors broadcastable to
``(batch, num_heads, query_len, key_len)``, where ``True`` means "may attend"
and ``False`` means "masked out". ``Transformer.forward`` builds the source
padding mask and the target padding + causal mask from ``pad_idx`` unless
masks are passed explicitly.
"""

from __future__ import annotations

import copy
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

__all__ = [
    "scaled_dot_product_attention",
    "make_padding_mask",
    "make_causal_mask",
    "MultiHeadAttention",
    "PositionwiseFeedForward",
    "PositionalEncoding",
    "EncoderLayer",
    "DecoderLayer",
    "Encoder",
    "Decoder",
    "Transformer",
]


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Module] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute ``softmax(Q K^T / sqrt(d_k)) V``.

    Args:
        query: ``(..., q_len, d_k)``
        key:   ``(..., k_len, d_k)``
        value: ``(..., k_len, d_v)``
        mask:  optional boolean tensor broadcastable to ``(..., q_len, k_len)``;
            ``True`` keeps a position, ``False`` masks it out.
        dropout: optional dropout module applied to the attention weights.

    Returns:
        ``(output, weights)`` with shapes ``(..., q_len, d_v)`` and
        ``(..., q_len, k_len)``. Masked positions receive exactly zero weight.
    """
    d_k = query.size(-1)
    # Scale by sqrt(d_k) *before* the softmax.
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # Use the most negative finite value rather than -inf so that a row in
        # which every key is masked yields a uniform distribution instead of NaN.
        scores = scores.masked_fill(~mask.to(torch.bool), torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    if mask is not None:
        # Guarantee exact zeros at masked positions.
        weights = weights.masked_fill(~mask.to(torch.bool), 0.0)
    if dropout is not None:
        weights = dropout(weights)
    return torch.matmul(weights, value), weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention with separate Q/K/V and output linear projections.

    ``d_model`` is split evenly into ``num_heads`` heads of size
    ``d_k = d_model // num_heads``. The attention weights from the most recent
    forward pass are stored in ``self.attn_weights`` with shape
    ``(batch, num_heads, q_len, k_len)`` for inspection.
    """

    def __init__(self, d_model: int = 512, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights: Optional[torch.Tensor] = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, len, d_model) -> (batch, heads, len, d_k)
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args are ``(batch, len, d_model)``; returns ``(batch, q_len, d_model)``.

        ``mask`` must be broadcastable to ``(batch, num_heads, q_len, k_len)``.
        """
        batch = query.size(0)
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        out, weights = scaled_dot_product_attention(q, k, v, mask, self.dropout)
        self.attn_weights = weights.detach()

        # (batch, heads, q_len, d_k) -> (batch, q_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
        return self.w_o(out)


# ---------------------------------------------------------------------------
# Feed-forward and positional encoding
# ---------------------------------------------------------------------------
class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network: ``Linear -> ReLU -> Linear``."""

    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class PositionalEncoding(nn.Module):
    """Fixed (non-learned) sinusoidal positional encoding followed by dropout.

    ``PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))``
    ``PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))``
    """

    def __init__(self, d_model: int = 512, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])  # handles odd d_model
        # Non-persistent buffer: moved with .to()/.cuda(), never trained, and
        # not stored in the state dict (it is recomputed deterministically).
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(batch, len, d_model)`` -> same shape."""
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"sequence length {seq_len} exceeds PositionalEncoding max_len {self.pe.size(1)}"
            )
        x = x + self.pe[:, :seq_len].to(dtype=x.dtype)
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Encoder / decoder layers
# ---------------------------------------------------------------------------
class EncoderLayer(nn.Module):
    """Encoder layer: self-attention and feed-forward sublayers.

    Each sublayer uses post-LN: ``LayerNorm(x + Dropout(Sublayer(x)))``.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``x``: ``(batch, src_len, d_model)``; ``src_mask`` broadcastable to
        ``(batch, heads, src_len, src_len)``."""
        x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout2(self.feed_forward(x)))
        return x


class DecoderLayer(nn.Module):
    """Decoder layer: masked self-attention, cross-attention over the encoder
    output, and feed-forward sublayers, each with residual + post-LayerNorm."""

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``x``: ``(batch, tgt_len, d_model)``; ``memory``: encoder output
        ``(batch, src_len, d_model)``; ``tgt_mask`` (padding + causal) broadcastable
        to ``(batch, heads, tgt_len, tgt_len)``; ``src_mask`` broadcastable to
        ``(batch, heads, tgt_len, src_len)``."""
        x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout2(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.dropout3(self.feed_forward(x)))
        return x


# ---------------------------------------------------------------------------
# Stacks
# ---------------------------------------------------------------------------
def _clones(module: nn.Module, n: int) -> nn.ModuleList:
    """Return ``n`` independent deep copies of ``module``."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class Encoder(nn.Module):
    """Stack of ``num_layers`` identical (independently parameterised) encoder layers."""

    def __init__(
        self,
        num_layers: int = 6,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        layer = EncoderLayer(d_model, num_heads, d_ff, dropout, layer_norm_eps)
        self.layers = _clones(layer, num_layers)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    """Stack of ``num_layers`` identical (independently parameterised) decoder layers."""

    def __init__(
        self,
        num_layers: int = 6,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        layer = DecoderLayer(d_model, num_heads, d_ff, dropout, layer_norm_eps)
        self.layers = _clones(layer, num_layers)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------
def make_padding_mask(tokens: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """Boolean key-padding mask of shape ``(batch, 1, 1, len)``; ``True`` = real token."""
    return (tokens != pad_idx).unsqueeze(1).unsqueeze(2)


def make_causal_mask(size: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Lower-triangular look-ahead mask of shape ``(1, 1, size, size)``;
    position ``i`` may attend to positions ``<= i``."""
    mask = torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class Transformer(nn.Module):
    """Encoder-decoder Transformer (Vaswani et al., 2017).

    Token embeddings are scaled by ``sqrt(d_model)`` and summed with sinusoidal
    positional encodings. The decoder output is projected to target-vocabulary
    logits. Defaults follow the paper's base model.

    Args:
        src_vocab_size: source vocabulary size.
        tgt_vocab_size: target vocabulary size.
        d_model: model / embedding dimension (default 512).
        num_heads: attention heads (default 8).
        d_ff: inner feed-forward dimension (default 2048).
        num_layers: layers in each of encoder and decoder (default 6).
        num_decoder_layers: optional override for the decoder depth
            (defaults to ``num_layers``).
        dropout: dropout rate (default 0.1).
        max_len: maximum sequence length for positional encodings (default 5000).
        pad_idx: padding token id used to build padding masks (default 0).
        layer_norm_eps: LayerNorm epsilon (default 1e-5).
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_layers: int = 6,
        num_decoder_layers: Optional[int] = None,
        dropout: float = 0.1,
        max_len: int = 5000,
        pad_idx: int = 0,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if num_decoder_layers is None:
            num_decoder_layers = num_layers
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.src_pos = PositionalEncoding(d_model, dropout, max_len)
        self.tgt_pos = PositionalEncoding(d_model, dropout, max_len)

        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout, layer_norm_eps)
        self.decoder = Decoder(
            num_decoder_layers, d_model, num_heads, d_ff, dropout, layer_norm_eps
        )
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Xavier-uniform initialisation for all weight matrices."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Keep padding embeddings at zero after re-initialisation.
        with torch.no_grad():
            self.src_embed.weight[self.pad_idx].zero_()
            self.tgt_embed.weight[self.pad_idx].zero_()

    # -- masks -------------------------------------------------------------
    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Source padding mask ``(batch, 1, 1, src_len)``."""
        return make_padding_mask(src, self.pad_idx)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """Target padding + causal mask ``(batch, 1, tgt_len, tgt_len)``."""
        pad_mask = make_padding_mask(tgt, self.pad_idx)  # (batch, 1, 1, tgt_len)
        causal = make_causal_mask(tgt.size(1), device=tgt.device)  # (1, 1, tgt_len, tgt_len)
        return pad_mask & causal

    # -- forward -----------------------------------------------------------
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode ``src`` ``(batch, src_len)`` -> ``(batch, src_len, d_model)``."""
        if src_mask is None:
            src_mask = self.make_src_mask(src)
        x = self.src_pos(self.src_embed(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode ``tgt`` ``(batch, tgt_len)`` against encoder ``memory``;
        returns hidden states ``(batch, tgt_len, d_model)``."""
        if tgt_mask is None:
            tgt_mask = self.make_tgt_mask(tgt)
        y = self.tgt_pos(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        return self.decoder(y, memory, src_mask, tgt_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the full model.

        Args:
            src: source token ids ``(batch, src_len)``.
            tgt: target token ids ``(batch, tgt_len)``.
            src_mask: optional boolean mask broadcastable to
                ``(batch, 1, 1, src_len)``; built from ``pad_idx`` if omitted.
            tgt_mask: optional boolean mask broadcastable to
                ``(batch, 1, tgt_len, tgt_len)``; built from ``pad_idx`` plus a
                causal mask if omitted.

        Returns:
            Logits of shape ``(batch, tgt_len, tgt_vocab_size)``.
        """
        if src_mask is None:
            src_mask = self.make_src_mask(src)
        memory = self.encode(src, src_mask)
        hidden = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.generator(hidden)
