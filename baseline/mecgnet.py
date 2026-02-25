"""PyTorch implementation of mEcgNet baseline model.

Frequency-based segment partitioning with cascaded EcgModules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


class ConvBlock2D(nn.Module):
    """2D convolution block: Conv2D -> BatchNorm -> ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (2, 4),
        stride: Tuple[int, int] = (2, 2),
        padding: Tuple[int, int] = (0, 1),
        use_bn: bool = True,
        use_relu: bool = True,
        transpose: bool = False,
    ):
        super().__init__()
        self.use_bn = use_bn
        self.use_relu = use_relu

        if transpose:
            self.conv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding
            )
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding
            )

        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if use_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_relu:
            x = self.relu(x)
        return x


class MLPBlock(nn.Module):
    """MLP block for EcgModule-S.

    Contains Grouped Conv2D, Linear layer, BatchNorm, GELU, and Dropout.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        kernel_size: Tuple[int, int] = (1, 4),
        stride: Tuple[int, int] = (1, 4),
        groups: int = 64,
        padding: Tuple[int, int] = (0, 1),
        dropout: float = 0.1,
        is_encoder: bool = True,
    ):
        super().__init__()
        self.is_encoder = is_encoder

        if is_encoder:
            self.linear = nn.Linear(in_channels, hidden_dim)
            self.gconv = nn.Conv2d(
                hidden_dim, out_channels, kernel_size, stride,
                padding=padding, groups=min(groups, hidden_dim)
            )
        else:
            self.gconv = nn.ConvTranspose2d(
                in_channels, hidden_dim, (1, 1), (1, 1),
                padding=(0, 0), groups=min(groups, in_channels)
            )
            self.linear = nn.Linear(hidden_dim, out_channels)

        self.bn = nn.BatchNorm2d(out_channels if is_encoder else hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_encoder:
            x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
            x = self.linear(x)
            x = self.gelu(x)
            x = self.dropout(x)
            x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
            x = self.gconv(x)
            x = self.bn(x)
            x = self.gelu(x)
        else:
            x = self.gconv(x)
            x = self.bn(x)
            x = self.gelu(x)
            x = self.dropout(x)
            x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
            x = self.linear(x)
            x = self.gelu(x)
            x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        return x


class EcgModuleL(nn.Module):
    """EcgModule-L (Large Module).

    U-Net architecture processing (B, C, 12, 512) inputs to (B, 1, 12, 512) outputs.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        encoder_channels: List[int] = [64, 128, 256],
        decoder_channels: List[int] = [256, 128, 64],
        kernel_size: Tuple[int, int] = (2, 4),
        stride: Tuple[int, int] = (2, 2),
        padding: Tuple[int, int] = (0, 1),
    ):
        super().__init__()
        self.out_channels = out_channels

        # Encoder
        self.encoders = nn.ModuleList()
        ch_in = in_channels
        for i, ch_out in enumerate(encoder_channels):
            self.encoders.append(
                ConvBlock2D(ch_in, ch_out, kernel_size, stride, padding, use_bn=True, use_relu=True)
            )
            ch_in = ch_out

        # Decoder
        self.decoders = nn.ModuleList()
        for i, ch_out in enumerate(decoder_channels):
            if i == 0:
                dec_kernel = (3, 2)
                dec_padding = (0, 0)
            else:
                dec_kernel = kernel_size
                dec_padding = padding

            is_last = (i == len(decoder_channels) - 1)
            self.decoders.append(
                ConvBlock2D(
                    ch_in, ch_out, dec_kernel, stride, dec_padding,
                    use_bn=not is_last, use_relu=not is_last, transpose=True
                )
            )
            ch_in = ch_out

        # Final output layer
        self.output_conv = nn.ConvTranspose2d(
            decoder_channels[-1], out_channels, kernel_size, stride, padding
        )

    def forward(self, x: torch.Tensor, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input, shape (B, C, H, W) where H=12, W=512.
            target_size: Target output size (H, W) for alignment.

        Returns:
            Output, shape (B, 1, H, W).
        """
        if target_size is None:
            target_size = x.shape[2:]

        for encoder in self.encoders:
            x = encoder(x)
        for decoder in self.decoders:
            x = decoder(x)

        x = self.output_conv(x)

        if x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x


class EcgModuleS(nn.Module):
    """EcgModule-S (Small Module).

    Similar to EcgModule-L but replaces some conv blocks with MLP blocks.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        encoder_channels: List[int] = [64, 128, 256],
        decoder_channels: List[int] = [256, 128, 64],
        kernel_size: Tuple[int, int] = (2, 4),
        stride: Tuple[int, int] = (2, 2),
        padding: Tuple[int, int] = (0, 1),
        mlp_kernel: Tuple[int, int] = (1, 4),
        mlp_stride: Tuple[int, int] = (1, 4),
        mlp_groups: int = 64,
        mlp_dropout: float = 0.1,
    ):
        super().__init__()
        self.out_channels = out_channels

        # Encoder conv blocks (all except last)
        self.encoders = nn.ModuleList()
        ch_in = in_channels
        for i, ch_out in enumerate(encoder_channels[:-1]):
            self.encoders.append(
                ConvBlock2D(ch_in, ch_out, kernel_size, stride, padding, use_bn=True, use_relu=True)
            )
            ch_in = ch_out

        # Encoder last block: MLP
        self.encoder_mlp = MLPBlock(
            encoder_channels[-2], encoder_channels[-1],
            hidden_dim=encoder_channels[-1],
            kernel_size=mlp_kernel, stride=mlp_stride,
            groups=mlp_groups, dropout=mlp_dropout, is_encoder=True
        )

        # Decoder first block: MLP
        self.decoder_mlp = MLPBlock(
            encoder_channels[-1], decoder_channels[0],
            hidden_dim=decoder_channels[0],
            kernel_size=(1, 1), stride=(1, 1),
            groups=4, dropout=mlp_dropout, is_encoder=False
        )

        # Decoder conv blocks (all except first)
        self.decoders = nn.ModuleList()
        ch_in = decoder_channels[0]
        for i, ch_out in enumerate(decoder_channels[1:]):
            is_last = (i == len(decoder_channels) - 2)
            self.decoders.append(
                ConvBlock2D(
                    ch_in, ch_out, kernel_size, stride, padding,
                    use_bn=not is_last, use_relu=not is_last, transpose=True
                )
            )
            ch_in = ch_out

        # Final output layer
        self.output_conv = nn.ConvTranspose2d(
            decoder_channels[-1], out_channels, kernel_size, stride, padding
        )

    def forward(self, x: torch.Tensor, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input, shape (B, C, H, W) where H=12, W=512.
            target_size: Target output size (H, W).

        Returns:
            Output, shape (B, 1, H, W).
        """
        if target_size is None:
            target_size = x.shape[2:]

        for encoder in self.encoders:
            x = encoder(x)
        x = self.encoder_mlp(x)
        x = self.decoder_mlp(x)
        for decoder in self.decoders:
            x = decoder(x)

        x = self.output_conv(x)

        if x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x


class FrequencySegmenter(nn.Module):
    """Frequency segment partitioning module.

    Splits input signal into low, mid, and high frequency components via FFT.
    """

    def __init__(
        self,
        signal_length: int = 512,
        low_freq_ratio: float = 0.1,
        mid_freq_ratio: float = 0.3,
    ):
        super().__init__()
        self.signal_length = signal_length
        self.low_freq_ratio = low_freq_ratio
        self.mid_freq_ratio = mid_freq_ratio

        freq_bins = signal_length // 2 + 1
        self.low_end = int(freq_bins * low_freq_ratio)
        self.mid_end = int(freq_bins * (low_freq_ratio + mid_freq_ratio))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split signal into three frequency bands.

        Args:
            x: Input Lead-I signal, shape (B, 1, L).

        Returns:
            low_freq, mid_freq, high_freq: Time-domain signals, each (B, 1, L).
        """
        x_fft = torch.fft.rfft(x, dim=-1)
        freq_bins = x_fft.shape[-1]

        # Low frequency band
        low_mask = torch.zeros(freq_bins, device=x.device)
        low_mask[:self.low_end] = 1.0
        low_freq = torch.fft.irfft(x_fft * low_mask, n=self.signal_length, dim=-1)

        # Mid frequency band
        mid_mask = torch.zeros(freq_bins, device=x.device)
        mid_mask[self.low_end:self.mid_end] = 1.0
        mid_freq = torch.fft.irfft(x_fft * mid_mask, n=self.signal_length, dim=-1)

        # High frequency band
        high_mask = torch.zeros(freq_bins, device=x.device)
        high_mask[self.mid_end:] = 1.0
        high_freq = torch.fft.irfft(x_fft * high_mask, n=self.signal_length, dim=-1)

        return low_freq, mid_freq, high_freq


class MEcgNet(nn.Module):
    """mEcgNet: frequency-based 12-lead ECG reconstruction model.

    Pipeline:
        1. Split Lead-I (B, 1, 512) into low/mid/high frequency components.
        2. Replicate each component 12 times to form (B, 1, 12, 512) 2D representation.
        3. Cascaded EcgModule processing:
           - EcgModule-S (low): (B,1,12,512) -> (B,1,12,512)
           - EcgModule-S (mid): (B,2,12,512) -> (B,1,12,512)
           - EcgModule-L (high): (B,2,12,512) -> (B,1,12,512)
        4. Squeeze to (B, 12, 512).

    Input: Lead-I signal (B, 1, 512).
    Output: 12-lead ECG signal (B, 12, 512).
    """

    def __init__(self, config: dict):
        super().__init__()

        model_config = config.get('model', {})
        data_config = config.get('data', {})

        self.signal_length = data_config.get('signal_length', 512)
        self.num_leads = data_config.get('num_leads', 12)

        # Frequency segmenter
        self.freq_segmenter = FrequencySegmenter(
            signal_length=self.signal_length,
            low_freq_ratio=model_config.get('low_freq_ratio', 0.1),
            mid_freq_ratio=model_config.get('mid_freq_ratio', 0.3),
        )

        ecgmodule_l_config = model_config.get('ecgmodule_l', {})
        ecgmodule_s_config = model_config.get('ecgmodule_s', {})

        # EcgModule-S for low frequency (first module, in_channels=1)
        self.module_low = self._build_module_s(ecgmodule_s_config, in_channels=1, out_channels=1)
        # EcgModule-S for mid frequency (second module, in_channels=2)
        self.module_mid = self._build_module_s(ecgmodule_s_config, in_channels=2, out_channels=1)
        # EcgModule-L for high frequency (third module, in_channels=2)
        self.module_high = self._build_module_l(ecgmodule_l_config, in_channels=2, out_channels=1)

    def _build_module_s(self, config: dict, in_channels: int, out_channels: int) -> nn.Module:
        """Build EcgModule-S."""
        return EcgModuleS(
            in_channels=in_channels,
            out_channels=out_channels,
            encoder_channels=config.get('encoder_channels', [64, 128, 256]),
            decoder_channels=config.get('decoder_channels', [256, 128, 64]),
            kernel_size=tuple(config.get('kernel_size', [2, 4])),
            stride=tuple(config.get('stride', [2, 2])),
            padding=tuple(config.get('padding', [0, 1])),
            mlp_kernel=tuple(config.get('mlp_kernel', [1, 4])),
            mlp_stride=tuple(config.get('mlp_stride', [1, 4])),
            mlp_groups=config.get('mlp_groups', 64),
            mlp_dropout=config.get('mlp_dropout', 0.1),
        )

    def _build_module_l(self, config: dict, in_channels: int, out_channels: int) -> nn.Module:
        """Build EcgModule-L."""
        return EcgModuleL(
            in_channels=in_channels,
            out_channels=out_channels,
            encoder_channels=config.get('encoder_channels', [64, 128, 256]),
            decoder_channels=config.get('decoder_channels', [256, 128, 64]),
            kernel_size=tuple(config.get('kernel_size', [2, 4])),
            stride=tuple(config.get('stride', [2, 2])),
            padding=tuple(config.get('padding', [0, 1])),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Lead-I input signal, shape (B, 1, L).

        Returns:
            Reconstructed 12-lead ECG, shape (B, 12, L).
        """
        target_size = (self.num_leads, self.signal_length)

        # Frequency decomposition
        low_freq, mid_freq, high_freq = self.freq_segmenter(x)

        # Replicate to 2D representation: (B, 1, L) -> (B, 1, 12, L)
        low_freq_2d = low_freq.repeat(1, self.num_leads, 1).unsqueeze(1)
        mid_freq_2d = mid_freq.repeat(1, self.num_leads, 1).unsqueeze(1)
        high_freq_2d = high_freq.repeat(1, self.num_leads, 1).unsqueeze(1)

        # Cascaded processing
        out_low = self.module_low(low_freq_2d, target_size=target_size)

        mid_input = torch.cat([mid_freq_2d, out_low], dim=1)
        out_mid = self.module_mid(mid_input, target_size=target_size)

        high_input = torch.cat([high_freq_2d, out_mid], dim=1)
        out_high = self.module_high(high_input, target_size=target_size)

        output = out_high.squeeze(1)  # (B, 12, L)
        return output


def build_mecgnet(config: dict) -> MEcgNet:
    """Build mEcgNet model from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        MEcgNet model instance.
    """
    return MEcgNet(config)
