"""Meta Tokenizer: Embeds patient metadata into semantic tokens.

Processes heterogeneous clinical metadata (age, sex, height, weight) into
a unified sequence of high-dimensional tokens. Continuous attributes use
sinusoidal positional encoding followed by MLP projection; discrete
attributes use learnable embedding lookup tables. Missing or invalid
values are handled via learnable null tokens.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class SinusoidalPositionEncoding(nn.Module):
    """Sinusoidal encoding for continuous scalar values."""

    def __init__(self, embed_dim: int, max_period: float = 10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        half_dim = embed_dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim)
        self.register_buffer('freqs', freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        args = x * self.freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.embed_dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ContinuousEmbedding(nn.Module):
    """Embedding module for continuous attributes via sinusoidal encoding + MLP."""

    def __init__(self, sin_embed_dim: int = 64, hidden_dim: int = 256,
                 output_dim: int = 256, num_layers: int = 2,
                 dropout: float = 0.1,
                 value_range: Tuple[float, float] = (0, 100),
                 max_period: float = 10000.0):
        super().__init__()
        self.value_range = value_range
        self.sin_embed = SinusoidalPositionEncoding(sin_embed_dim, max_period)

        layers = []
        in_dim = sin_embed_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)
        self.null_embedding = nn.Parameter(torch.randn(output_dim) * 0.02)

    def is_valid(self, x: torch.Tensor,
                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        valid = (x >= self.value_range[0]) & (x <= self.value_range[1]) & (~torch.isnan(x))
        if mask is not None:
            valid = valid & mask
        return valid

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.shape[0]
        valid_mask = self.is_valid(x, mask)

        safe_value = self.value_range[0]
        x_safe = torch.nan_to_num(x, nan=safe_value).clamp(
            self.value_range[0], self.value_range[1])
        x_normalized = (x_safe - self.value_range[0]) / (
            self.value_range[1] - self.value_range[0] + 1e-8)
        x_normalized = x_normalized.clamp(0, 1)

        sin_embed = self.sin_embed(x_normalized)
        embedding = self.mlp(sin_embed)

        null_embed = self.null_embedding.unsqueeze(0).expand(batch_size, -1)
        embedding = torch.where(
            valid_mask.unsqueeze(-1).expand(-1, embedding.shape[-1]),
            embedding, null_embed)
        return embedding


class DiscreteEmbedding(nn.Module):
    """Embedding module for discrete attributes via learnable lookup table."""

    def __init__(self, num_categories: int, embed_dim: int = 256,
                 include_null: bool = True):
        super().__init__()
        self.num_categories = num_categories
        self.include_null = include_null
        total = num_categories + (1 if include_null else 0)
        self.embedding = nn.Embedding(total, embed_dim)
        self.null_idx = num_categories if include_null else None
        nn.init.normal_(self.embedding.weight, std=0.02)

    def is_valid(self, x: torch.Tensor,
                 mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        valid = (x >= 0) & (x < self.num_categories)
        if mask is not None:
            valid = valid & mask
        return valid

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        valid_mask = self.is_valid(x, mask)
        x_safe = x.clone()
        if self.include_null:
            x_safe = torch.where(valid_mask, x, torch.full_like(x, self.null_idx))
        else:
            x_safe = torch.where(valid_mask, x, torch.zeros_like(x))
        return self.embedding(x_safe)


class MetaTokenizer(nn.Module):
    """Transforms patient metadata into a sequence of semantic tokens.

    Handles continuous variables (age, height, weight) via sinusoidal
    encoding + MLP, and discrete variables (sex) via embedding lookup.
    Supports learnable null tokens for missing values and modality dropout.
    """

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 512,
                 output_dim: int = 256, sin_embed_dim: int = 64,
                 max_period: float = 10000.0,
                 age_range: Tuple[float, float] = (0, 100),
                 height_range: Tuple[float, float] = (100, 220),
                 weight_range: Tuple[float, float] = (20, 200),
                 num_sex_categories: int = 3,
                 mlp_layers: int = 2, mlp_dropout: float = 0.1,
                 num_tokens: int = 4):
        super().__init__()
        self.output_dim = output_dim
        self.num_tokens = num_tokens

        self.age_embed = ContinuousEmbedding(
            sin_embed_dim, hidden_dim, output_dim, mlp_layers,
            mlp_dropout, age_range, max_period)
        self.sex_embed = DiscreteEmbedding(
            num_sex_categories - 1, output_dim, include_null=True)
        self.height_embed = ContinuousEmbedding(
            sin_embed_dim, hidden_dim, output_dim, mlp_layers,
            mlp_dropout, height_range, max_period)
        self.weight_embed = ContinuousEmbedding(
            sin_embed_dim, hidden_dim, output_dim, mlp_layers,
            mlp_dropout, weight_range, max_period)

        self.global_null_embedding = nn.Parameter(
            torch.randn(num_tokens, output_dim) * 0.02)
        self.token_type_embed = nn.Embedding(num_tokens, output_dim)
        nn.init.normal_(self.token_type_embed.weight, std=0.02)

    def forward(self, age, sex, height, weight,
                age_mask=None, sex_mask=None,
                height_mask=None, weight_mask=None,
                drop_hw_prob: float = 0.0, uncond_prob: float = 0.0,
                force_uncond: bool = False) -> Dict[str, torch.Tensor]:
        batch_size = age.shape[0]
        device = age.device

        if force_uncond:
            tokens = self.global_null_embedding.unsqueeze(0).expand(batch_size, -1, -1)
            return {'tokens': tokens,
                    'is_uncond': torch.ones(batch_size, dtype=torch.bool, device=device)}

        is_uncond = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if uncond_prob > 0 and self.training:
            is_uncond = torch.rand(batch_size, device=device) < uncond_prob

        hw_drop_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if drop_hw_prob > 0 and self.training:
            hw_drop_mask = torch.rand(batch_size, device=device) < drop_hw_prob

        age_emb = self.age_embed(age, age_mask)
        sex_emb = self.sex_embed(sex.long(), sex_mask)
        height_emb = self.height_embed(height, height_mask)
        weight_emb = self.weight_embed(weight, weight_mask)

        if hw_drop_mask.any():
            null_h = self.height_embed.null_embedding.unsqueeze(0).expand(batch_size, -1)
            null_w = self.weight_embed.null_embedding.unsqueeze(0).expand(batch_size, -1)
            mask_expanded = hw_drop_mask.unsqueeze(-1).expand_as(height_emb)
            height_emb = torch.where(mask_expanded, null_h, height_emb)
            weight_emb = torch.where(mask_expanded, null_w, weight_emb)

        tokens = torch.stack([age_emb, sex_emb, height_emb, weight_emb], dim=1)
        token_type_ids = torch.arange(self.num_tokens, device=device)
        tokens = tokens + self.token_type_embed(token_type_ids).unsqueeze(0)

        if is_uncond.any():
            global_null = self.global_null_embedding.unsqueeze(0).expand(batch_size, -1, -1)
            tokens = torch.where(
                is_uncond.view(-1, 1, 1).expand_as(tokens), global_null, tokens)

        return {'tokens': tokens, 'is_uncond': is_uncond}

    def get_null_embedding(self, batch_size: int,
                           device: torch.device) -> torch.Tensor:
        return self.global_null_embedding.unsqueeze(0).expand(batch_size, -1, -1)


class MetaTokenizerWrapper(nn.Module):
    """Wrapper that accepts dictionary-format metadata input."""

    def __init__(self, config: dict):
        super().__init__()
        mt_config = config.get('meta_tokenizer', {})
        self.tokenizer = MetaTokenizer(
            embed_dim=mt_config.get('embed_dim', 256),
            hidden_dim=mt_config.get('hidden_dim', 512),
            output_dim=mt_config.get('output_dim', 256),
            sin_embed_dim=mt_config.get('sin_embed_dim', 64),
            max_period=mt_config.get('max_period', 10000.0),
            age_range=tuple(mt_config.get('age_range', [0, 100])),
            height_range=tuple(mt_config.get('height_range', [100, 220])),
            weight_range=tuple(mt_config.get('weight_range', [20, 200])),
            num_sex_categories=mt_config.get('num_sex_categories', 3),
            mlp_layers=mt_config.get('mlp_layers', 2),
            mlp_dropout=mt_config.get('mlp_dropout', 0.1),
            num_tokens=mt_config.get('num_tokens', 4))
        self.output_dim = mt_config.get('output_dim', 256)
        self.num_tokens = mt_config.get('num_tokens', 4)

    def forward(self, metadata: Dict[str, torch.Tensor],
                drop_hw_prob: float = 0.0, uncond_prob: float = 0.0,
                force_uncond: bool = False) -> Dict[str, torch.Tensor]:
        age = metadata.get('age', torch.zeros(1))
        sex = metadata.get('sex', torch.zeros(1))
        height = metadata.get('height', torch.zeros(1))
        weight = metadata.get('weight', torch.zeros(1))

        age_mask = ~torch.isnan(age) if age.is_floating_point() else None
        height_mask = ~torch.isnan(height) if height.is_floating_point() else None
        weight_mask = ~torch.isnan(weight) if weight.is_floating_point() else None

        if sex.is_floating_point():
            sex_mask = ~torch.isnan(sex) & (sex >= 0)
            sex_safe = torch.where(sex_mask, sex, torch.zeros_like(sex)).long()
        else:
            sex_safe = sex.long()
            sex_mask = (sex_safe >= 0)

        return self.tokenizer(
            age=age, sex=sex_safe, height=height, weight=weight,
            age_mask=age_mask, sex_mask=sex_mask,
            height_mask=height_mask, weight_mask=weight_mask,
            drop_hw_prob=drop_hw_prob, uncond_prob=uncond_prob,
            force_uncond=force_uncond)
