"""PyTorch implementation of ECGRecover baseline model.

Hybrid 1D-2D convolutional autoencoder with skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


class Convolution1DLayer(nn.Module):
    """1D convolution layer applied independently per lead.

    Conv1D -> BatchNorm -> LeakyReLU -> Dropout.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        dropout: float = 0.2,
        leaky_slope: float = 0.02,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape (B, C, 12, L).

        Returns:
            Output tensor, shape (B, C_out, 12, L//2).
        """
        batch_size, channels, num_leads, seq_len = x.shape
        new_seq_len = seq_len // 2

        output = torch.zeros(
            batch_size, self.out_channels, num_leads, new_seq_len,
            device=x.device, dtype=x.dtype
        )
        for i in range(num_leads):
            output[:, :, i, :] = self.conv(x[:, :, i, :])
        return output


class Deconvolution1DLayer(nn.Module):
    """1D transposed convolution layer applied independently per lead.

    ConvTranspose1D -> BatchNorm -> LeakyReLU -> Dropout.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        dropout: float = 0.2,
        leaky_slope: float = 0.02,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(leaky_slope, inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape (B, C, 12, L).

        Returns:
            Output tensor, shape (B, C_out, 12, L*2).
        """
        batch_size, channels, num_leads, seq_len = x.shape
        new_seq_len = seq_len * 2

        output = torch.zeros(
            batch_size, self.out_channels, num_leads, new_seq_len,
            device=x.device, dtype=x.dtype
        )
        for i in range(num_leads):
            output[:, :, i, :] = self.deconv(x[:, :, i, :])
        return output


class Convolution2DLayer(nn.Module):
    """2D convolution layer: Conv2D -> BatchNorm -> LeakyReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (13, 4),
        stride: Tuple[int, int] = (1, 2),
        padding: Tuple[int, int] = (6, 1),
        leaky_slope: float = 0.02,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(leaky_slope, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Deconvolution2DLayer(nn.Module):
    """2D transposed convolution layer: ConvTranspose2D -> BatchNorm -> LeakyReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (13, 4),
        stride: Tuple[int, int] = (1, 2),
        padding: Tuple[int, int] = (6, 1),
        leaky_slope: float = 0.02,
    ):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(leaky_slope, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)


class MSEPearsonLoss(nn.Module):
    """Combined MSE and Pearson correlation loss.

    Loss = MSE - alpha * Pearson.
    """

    def __init__(self, pearson_weight: float = 0.1):
        super().__init__()
        self.pearson_weight = pearson_weight

    def pearson_correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Pearson correlation coefficient.

        Args:
            x: Predictions, shape (B, C, L).
            y: Targets, shape (B, C, L).

        Returns:
            Correlation coefficients, shape (B, C).
        """
        x_mean = torch.mean(x, dim=-1, keepdim=True)
        y_mean = torch.mean(y, dim=-1, keepdim=True)

        x_std = torch.std(x, dim=-1, unbiased=False)
        y_std = torch.std(y, dim=-1, unbiased=False)

        cov = torch.mean((x - x_mean) * (y - y_mean), dim=-1)
        r = cov / (x_std * y_std + 1e-8)
        r = torch.nan_to_num(r, nan=0.0)
        return r

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute loss.

        Args:
            pred: Predictions, shape (B, C, L).
            target: Targets, shape (B, C, L).

        Returns:
            total_loss, mean_pearson, mean_mse.
        """
        mse = torch.mean(torch.square(pred - target), dim=-1)
        mse = torch.nan_to_num(mse, nan=0.0)

        pearson = self.pearson_correlation(pred, target)

        loss = mse - self.pearson_weight * pearson
        total_loss = torch.mean(loss)
        mean_pearson = torch.mean(pearson)
        mean_mse = torch.mean(mse)

        return total_loss, mean_pearson, mean_mse


class ECGRecover(nn.Module):
    """ECGRecover: hybrid 1D-2D convolutional autoencoder for 12-lead ECG reconstruction.

    Architecture:
        1. Encoder uses parallel 1D and 2D convolutions to capture intra-lead and inter-lead features.
        2. Decoder uses skip connections to fuse multi-scale features.
        3. Output is activated by Tanh.

    Input: Lead-I signal (B, 1, 512).
    Output: 12-lead ECG signal (B, 12, 512).
    """

    def __init__(self, config: dict):
        super().__init__()

        model_config = config.get('model', {})
        data_config = config.get('data', {})

        self.signal_length = data_config.get('signal_length', 512)
        self.num_leads = data_config.get('num_leads', 12)

        encoder_channels = model_config.get('encoder_channels', [16, 32, 64, 128])
        dropout = model_config.get('dropout', 0.2)
        leaky_slope = model_config.get('leaky_relu_slope', 0.02)

        kernel_size_1d = model_config.get('kernel_size_1d', 4)
        stride_1d = model_config.get('stride_1d', 2)
        padding_1d = model_config.get('padding_1d', 1)

        kernel_size_2d = tuple(model_config.get('kernel_size_2d', [13, 4]))
        stride_2d = tuple(model_config.get('stride_2d', [1, 2]))
        padding_2d = tuple(model_config.get('padding_2d', [6, 1]))

        # ==================== Encoder ====================
        self.conv1d_1 = Convolution1DLayer(1, encoder_channels[0], kernel_size_1d, stride_1d, padding_1d, dropout, leaky_slope)
        self.conv2d_1 = Convolution2DLayer(1, encoder_channels[0], kernel_size_2d, stride_2d, padding_2d, leaky_slope)

        self.conv1d_2 = Convolution1DLayer(encoder_channels[0], encoder_channels[1], kernel_size_1d, stride_1d, padding_1d, dropout, leaky_slope)
        self.conv2d_2 = Convolution2DLayer(encoder_channels[0], encoder_channels[1], kernel_size_2d, stride_2d, padding_2d, leaky_slope)

        self.conv1d_3 = Convolution1DLayer(encoder_channels[1], encoder_channels[2], kernel_size_1d, stride_1d, padding_1d, dropout, leaky_slope)
        self.conv2d_3 = Convolution2DLayer(encoder_channels[1], encoder_channels[2], kernel_size_2d, stride_2d, padding_2d, leaky_slope)

        self.conv1d_4 = Convolution1DLayer(encoder_channels[2], encoder_channels[3], kernel_size_1d, stride_1d, padding_1d, dropout, leaky_slope)
        self.conv2d_4 = Convolution2DLayer(encoder_channels[2], encoder_channels[3], kernel_size_2d, stride_2d, padding_2d, leaky_slope)

        # ==================== Transition ====================
        self.transition = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=(13, 3), stride=(1, 1), padding=(6, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(leaky_slope, inplace=True),
        )

        # ==================== Decoder ====================
        self.deconv2d_1 = Deconvolution2DLayer(256, 128, kernel_size_2d, stride_2d, padding_2d, leaky_slope)
        self.deconv2d_2 = Deconvolution2DLayer(256, 64, kernel_size_2d, stride_2d, padding_2d, leaky_slope)
        self.deconv2d_3 = Deconvolution2DLayer(128, 32, kernel_size_2d, stride_2d, padding_2d, leaky_slope)
        self.deconv2d_4 = Deconvolution2DLayer(64, 1, kernel_size_2d, stride_2d, padding_2d, leaky_slope)

        # ==================== Output ====================
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=(13, 3), stride=(1, 1), padding=(6, 1)),
            nn.Tanh(),
        )

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Replicate Lead-I to 12 leads.

        Args:
            x: Lead-I input, shape (B, 1, L).

        Returns:
            Expanded input, shape (B, 1, 12, L).
        """
        x_expanded = x.repeat(1, self.num_leads, 1)  # (B, 12, L)
        x_expanded = x_expanded.unsqueeze(1)  # (B, 1, 12, L)
        return x_expanded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Lead-I input signal, shape (B, 1, L).

        Returns:
            Reconstructed 12-lead ECG, shape (B, 12, L).
        """
        x = self._prepare_input(x)  # (B, 1, 12, L)

        # ==================== Encoder ====================
        conv1d_1 = self.conv1d_1(x)
        conv2d_1 = self.conv2d_1(x)
        conv_1 = torch.cat([conv1d_1, conv2d_1], dim=1)

        conv1d_2 = self.conv1d_2(conv1d_1)
        conv2d_2 = self.conv2d_2(conv2d_1)
        conv_2 = torch.cat([conv1d_2, conv2d_2], dim=1)

        conv1d_3 = self.conv1d_3(conv1d_2)
        conv2d_3 = self.conv2d_3(conv2d_2)
        conv_3 = torch.cat([conv1d_3, conv2d_3], dim=1)

        conv1d_4 = self.conv1d_4(conv1d_3)
        conv2d_4 = self.conv2d_4(conv2d_3)
        conv_4 = torch.cat([conv1d_4, conv2d_4], dim=1)

        # ==================== Transition ====================
        transition = self.transition(conv_4)

        # ==================== Decoder ====================
        deconv_1 = self.deconv2d_1(conv_4)
        deconv_1 = torch.cat([deconv_1, conv_3], dim=1)

        deconv_2 = self.deconv2d_2(deconv_1)
        deconv_2 = torch.cat([deconv_2, conv_2], dim=1)

        deconv_3 = self.deconv2d_3(deconv_2)
        deconv_3 = torch.cat([deconv_3, conv_1], dim=1)

        deconv_4 = self.deconv2d_4(deconv_3)

        # ==================== Output ====================
        output = self.final_conv(deconv_4)
        output = output.squeeze(1)  # (B, 12, L)
        return output


def build_ecgrecover(config: dict) -> ECGRecover:
    """Build ECGRecover model from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        ECGRecover model instance.
    """
    return ECGRecover(config)
