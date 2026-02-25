"""MetaFormer: Metadata-Guided Transformer for 12-Lead ECG Reconstruction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Union, Tuple

USE_METADATA = True

try:
    from model.meta_tokenizer import MetaTokenizerWrapper
except ImportError:
    try:
        from .meta_tokenizer import MetaTokenizerWrapper
    except ImportError:
        raise ImportError("Failed to import MetaTokenizer modules.")


# ========================= Loss Functions =========================

class MSEPearsonLoss(nn.Module):
    """Hybrid loss: MSE + weighted Pearson correlation loss."""

    def __init__(self, pearson_weight: float = 0.1):
        super().__init__()
        self.pearson_weight = pearson_weight

    def pearson_correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Pearson correlation along the temporal dimension."""
        x_mean = torch.mean(x, dim=-1, keepdim=True)
        y_mean = torch.mean(y, dim=-1, keepdim=True)
        x_std = torch.std(x, dim=-1, unbiased=False)
        y_std = torch.std(y, dim=-1, unbiased=False)
        cov = torch.mean((x - x_mean) * (y - y_mean), dim=-1)
        r = cov / (x_std * y_std + 1e-8)
        return torch.nan_to_num(r, nan=0.0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = F.mse_loss(pred, target)
        pearson_loss = 1.0 - torch.mean(self.pearson_correlation(pred, target))
        return mse_loss + self.pearson_weight * pearson_loss


# ========================= Building Blocks =========================

class GroupNorm32(nn.GroupNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


def zero_module(module: nn.Module) -> nn.Module:
    """Zero-initialize module parameters."""
    for p in module.parameters():
        p.detach().zero_()
    return module


class ResBlock1D(nn.Module):
    """1D residual block (MorphResBlock) with GroupNorm, SiLU, and dropout."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = GroupNorm32(32, in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = GroupNorm32(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip_conv = (nn.Conv1d(in_channels, out_channels, kernel_size=1)
                          if in_channels != out_channels else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip_conv(x)


# ========================= Attention Modules =========================

class SelfAttention1D(nn.Module):
    """Multi-head self-attention for temporal feature modeling."""

    def __init__(self, channels: int, num_heads: int = 8,
                 head_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.head_dim = head_dim or (channels // num_heads)
        self.inner_dim = num_heads * self.head_dim
        self.norm = GroupNorm32(32, channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.inner_dim, num_heads=num_heads,
            dropout=dropout, batch_first=False)
        self.in_proj = nn.Conv1d(channels, self.inner_dim, kernel_size=1)
        self.out_proj = nn.Sequential(
            nn.Conv1d(self.inner_dim, channels, kernel_size=1), nn.Dropout(dropout))
        zero_module(self.out_proj[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(self.norm(x)).permute(2, 0, 1)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        return x + self.out_proj(attn_out.permute(1, 2, 0))


class CrossAttention1D(nn.Module):
    """Multi-head cross-attention for metadata-guided feature recalibration.

    ECG features serve as queries; metadata tokens serve as keys and values.
    """

    def __init__(self, channels: int, context_dim: int, num_heads: int = 8,
                 head_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.head_dim = head_dim or (channels // num_heads)
        self.inner_dim = num_heads * self.head_dim
        self.norm = GroupNorm32(32, channels)
        self.norm_context = nn.LayerNorm(context_dim)
        self.q_proj = nn.Conv1d(channels, self.inner_dim, kernel_size=1)
        self.kv_proj = nn.Linear(context_dim, self.inner_dim * 2)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.inner_dim, num_heads=num_heads,
            dropout=dropout, batch_first=False,
            kdim=self.inner_dim, vdim=self.inner_dim)
        self.out_proj = nn.Sequential(
            nn.Conv1d(self.inner_dim, channels, kernel_size=1), nn.Dropout(dropout))
        zero_module(self.out_proj[0])

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        context = self.norm_context(context)
        q = self.q_proj(h).permute(2, 0, 1)
        kv = self.kv_proj(context)
        k, v = torch.chunk(kv, 2, dim=-1)
        k, v = k.permute(1, 0, 2), v.permute(1, 0, 2)
        attn_out, _ = self.attn(q, k, v, need_weights=False)
        return x + self.out_proj(attn_out.permute(1, 2, 0))


class TransformerBlock1D(nn.Module):
    """Metadata-Guided Transformer block: self-attention + cross-attention + FFN."""

    def __init__(self, channels: int, context_dim: int, num_heads: int = 8,
                 head_dim: Optional[int] = None, dropout: float = 0.0,
                 use_cross_attention: bool = True, use_self_attention: bool = True):
        super().__init__()
        self.use_cross_attention = use_cross_attention
        self.use_self_attention = use_self_attention

        if use_self_attention:
            self.self_attn = SelfAttention1D(channels, num_heads, head_dim, dropout)
        if use_cross_attention:
            self.cross_attn = CrossAttention1D(
                channels, context_dim, num_heads, head_dim, dropout)

        self.norm = GroupNorm32(32, channels)
        self.ffn = nn.Sequential(
            nn.Conv1d(channels, channels * 4, kernel_size=1), nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels * 4, channels, kernel_size=1), nn.Dropout(dropout))
        zero_module(self.ffn[-2])

    def forward(self, x: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.use_self_attention:
            x = self.self_attn(x)
        if self.use_cross_attention and context is not None and USE_METADATA:
            x = self.cross_attn(x, context)
        return x + self.ffn(self.norm(x))


# ========================= U-Net Architecture =========================

class DownBlock(nn.Module):
    """Encoder stage: residual blocks + optional attention + strided downsampling."""

    def __init__(self, in_channels: int, out_channels: int, context_dim: int,
                 num_res_blocks: int = 2, num_heads: int = 8, dropout: float = 0.1,
                 downsample: bool = True, use_attention: bool = True,
                 use_cross_attention: bool = True):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            ch_in = in_channels if i == 0 else out_channels
            self.res_blocks.append(ResBlock1D(ch_in, out_channels, dropout))
            self.attn_blocks.append(
                TransformerBlock1D(out_channels, context_dim, num_heads, dropout=dropout,
                                   use_cross_attention=use_cross_attention)
                if use_attention else nn.Identity())
        self.downsample = (nn.Conv1d(out_channels, out_channels, 3, stride=2, padding=1)
                           if downsample else None)

    def forward(self, x: torch.Tensor,
                context: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        hiddens = []
        for res_block, attn_block in zip(self.res_blocks, self.attn_blocks):
            x = res_block(x)
            if not isinstance(attn_block, nn.Identity):
                x = attn_block(x, context)
            hiddens.append(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x, hiddens


class UpBlock(nn.Module):
    """Decoder stage: skip connections + residual blocks + linear upsampling."""

    def __init__(self, in_channels: int, out_channels: int, context_dim: int,
                 num_res_blocks: int = 2, num_heads: int = 8, dropout: float = 0.1,
                 upsample: bool = True, use_attention: bool = True,
                 use_cross_attention: bool = True):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        self.attn_blocks = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.res_blocks.append(ResBlock1D(in_channels * 2, in_channels, dropout))
            self.attn_blocks.append(
                TransformerBlock1D(in_channels, context_dim, num_heads, dropout=dropout,
                                   use_cross_attention=use_cross_attention)
                if use_attention else nn.Identity())
        self.upsample = (nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
            if upsample else None)

    def forward(self, x: torch.Tensor, skip_connections: List[torch.Tensor],
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        for res_block, attn_block in zip(self.res_blocks, self.attn_blocks):
            if len(skip_connections) > 0:
                skip = skip_connections.pop()
                if x.shape[-1] != skip.shape[-1]:
                    x = F.interpolate(x, size=skip.shape[-1],
                                      mode='linear', align_corners=False)
                x = torch.cat([x, skip], dim=1)
            x = res_block(x)
            if not isinstance(attn_block, nn.Identity):
                x = attn_block(x, context)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class MiddleBlock(nn.Module):
    """Bottleneck with self-attention and metadata-guided cross-attention."""

    def __init__(self, channels: int, context_dim: int,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.res_block1 = ResBlock1D(channels, channels, dropout)
        self.attn = TransformerBlock1D(channels, context_dim, num_heads, dropout=dropout)
        self.res_block2 = ResBlock1D(channels, channels, dropout)

    def forward(self, x: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.res_block2(self.attn(self.res_block1(x), context))


class ReconstructionUNet(nn.Module):
    """U-Net backbone for direct ECG reconstruction from single lead."""

    def __init__(self, in_channels: int = 1, out_channels: int = 11,
                 in_length: int = 512, base_channels: int = 64,
                 channel_multipliers: List[int] = [1, 2, 4, 8],
                 num_res_blocks: int = 2,
                 attention_resolutions: List[int] = [32, 16, 8],
                 num_heads: int = 8, context_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.input_conv = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        channels = [base_channels * m for m in channel_multipliers]
        current_res = in_length

        ch_in = base_channels
        for i, ch_out in enumerate(channels):
            downsample = i < len(channels) - 1
            self.down_blocks.append(DownBlock(
                ch_in, ch_out, context_dim, num_res_blocks, num_heads, dropout,
                downsample, use_attention=(current_res in attention_resolutions)))
            ch_in = ch_out
            if downsample:
                current_res //= 2

        self.middle_block = MiddleBlock(channels[-1], context_dim, num_heads, dropout)

        reversed_channels = list(reversed(channels[:-1])) + [base_channels]
        ch_in = channels[-1]
        for i, ch_out in enumerate(reversed_channels):
            upsample = i < len(reversed_channels) - 1
            self.up_blocks.append(UpBlock(
                ch_in, ch_out, context_dim, num_res_blocks, num_heads, dropout,
                upsample, use_attention=(current_res in attention_resolutions)))
            ch_in = ch_out
            if upsample:
                current_res *= 2

        self.out_norm = GroupNorm32(32, base_channels)
        self.out_conv = nn.Conv1d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x_ref: torch.Tensor,
                meta_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.input_conv(x_ref)
        skip_connections = []
        for down_block in self.down_blocks:
            h, hiddens = down_block(h, meta_tokens)
            skip_connections.extend(hiddens)
        h = self.middle_block(h, meta_tokens)
        for up_block in self.up_blocks:
            h = up_block(h, skip_connections, meta_tokens)
        return self.out_conv(F.silu(self.out_norm(h)))


# ========================= MetaFormer =========================

class MetaFormer(nn.Module):
    """MetaFormer: Metadata-Guided Transformer for 12-Lead ECG Reconstruction.

    Reconstructs the remaining 11 leads from a single reference lead (Lead I)
    conditioned on patient metadata through cross-attention.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        model_config = config.get('model', {})
        data_config = config.get('data', {})

        self.num_leads = data_config.get('num_leads', 12)
        self.condition_leads = data_config.get('condition_leads', [0])
        self.target_leads = [i for i in range(self.num_leads)
                             if i not in self.condition_leads]
        self.num_target_leads = len(self.target_leads)
        self.signal_length = data_config.get('signal_length', 512)

        self.criterion = MSEPearsonLoss(pearson_weight=0.1)
        self.meta_tokenizer = MetaTokenizerWrapper(model_config)

        unet_config = model_config.get('unet', {})
        self.unet = ReconstructionUNet(
            in_channels=len(self.condition_leads),
            out_channels=self.num_target_leads,
            in_length=self.signal_length,
            base_channels=unet_config.get('base_channels', 64),
            channel_multipliers=unet_config.get('channel_multipliers', [1, 2, 4, 8]),
            num_res_blocks=unet_config.get('num_res_blocks', 2),
            attention_resolutions=unet_config.get('attention_resolutions', [32, 16, 8]),
            num_heads=unet_config.get('num_heads', 8),
            context_dim=model_config.get('meta_tokenizer', {}).get('output_dim', 256),
            dropout=unet_config.get('dropout', 0.1))

    def forward(self, x: torch.Tensor, metadata: Dict[str, torch.Tensor],
                drop_hw_prob: float = 0.0,
                uncond_prob: float = 0.0) -> Dict[str, torch.Tensor]:
        """Training forward pass.

        Args:
            x: 12-lead ECG signal, shape (B, 12, L).
            metadata: Patient metadata dictionary.
            drop_hw_prob: Modality dropout probability for height/weight.
            uncond_prob: Probability of dropping all metadata.

        Returns:
            Dictionary with 'loss', 'pred', and 'target'.
        """
        x_ref = x[:, self.condition_leads, :]
        x_target = x[:, self.target_leads, :]

        meta_output = self.meta_tokenizer(
            metadata, drop_hw_prob=drop_hw_prob,
            uncond_prob=uncond_prob, force_uncond=False)
        pred = self.unet(x_ref, meta_output['tokens'])
        loss = self.criterion(pred, x_target)

        return {'loss': loss, 'pred': pred, 'target': x_target}

    @torch.no_grad()
    def sample(self, x_ref: torch.Tensor, metadata: Dict[str, torch.Tensor],
               **kwargs) -> torch.Tensor:
        """Reconstruct target leads from reference lead.

        Args:
            x_ref: Reference lead, shape (B, 1, L).
            metadata: Patient metadata dictionary.

        Returns:
            Predicted target leads, shape (B, 11, L).
        """
        meta_output = self.meta_tokenizer(
            metadata, drop_hw_prob=0.0, uncond_prob=0.0, force_uncond=False)
        return self.unet(x_ref, meta_output['tokens'])

    def reconstruct_12lead(self, x_ref: torch.Tensor,
                           metadata: Dict[str, torch.Tensor],
                           **kwargs) -> torch.Tensor:
        """Reconstruct full 12-lead ECG from reference lead.

        Returns:
            Reconstructed 12-lead signal, shape (B, 12, L).
        """
        x_target = self.sample(x_ref, metadata, **kwargs)
        batch_size = x_ref.shape[0]
        x_full = torch.zeros(batch_size, self.num_leads, self.signal_length,
                             device=x_ref.device, dtype=x_ref.dtype)
        for i, lead_idx in enumerate(self.condition_leads):
            x_full[:, lead_idx, :] = x_ref[:, i, :]
        for i, lead_idx in enumerate(self.target_leads):
            x_full[:, lead_idx, :] = x_target[:, i, :]
        return x_full

def build_model(config: dict) -> MetaFormer:
    """Build MetaFormer model from configuration."""
    return MetaFormer(config)

