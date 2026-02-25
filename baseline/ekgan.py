"""PyTorch implementation of EKGAN baseline model.

Conditional GAN for 12-lead ECG reconstruction from Lead-I.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict


def weights_init_normal(m, mean: float = 0.0, std: float = 0.02):
    """Initialize weights with normal distribution."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, mean, std)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, std)
        nn.init.constant_(m.bias.data, 0.0)


def compute_same_padding(
    input_size: Tuple[int, int],
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
) -> Tuple[int, int, int, int]:
    """Compute TensorFlow-style 'same' padding.

    Args:
        input_size: (H, W).
        kernel_size: (kH, kW).
        stride: (sH, sW).

    Returns:
        (pad_left, pad_right, pad_top, pad_bottom).
    """
    h, w = input_size
    kh, kw = kernel_size
    sh, sw = stride

    out_h = (h + sh - 1) // sh
    out_w = (w + sw - 1) // sw

    pad_h = max((out_h - 1) * sh + kh - h, 0)
    pad_w = max((out_w - 1) * sw + kw - w, 0)

    # Distribute padding (TF-style: more on right/bottom)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    return (pad_left, pad_right, pad_top, pad_bottom)


class EncoderBlock2D(nn.Module):
    """2D encoder block: Conv2D -> BatchNorm -> LeakyReLU with dynamic same padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (2, 4),
        stride: Tuple[int, int] = (2, 2),
        use_bn: bool = True,
    ):
        super().__init__()
        self.use_bn = use_bn
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            padding=0, bias=False
        )
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        pad = compute_same_padding((h, w), self.kernel_size, self.stride)
        x = F.pad(x, pad, mode='constant', value=0)
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.activation(x)
        return x


class DecoderBlock2D(nn.Module):
    """2D decoder block: ConvTranspose2D -> BatchNorm -> ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (2, 4),
        stride: Tuple[int, int] = (2, 2),
        use_bn: bool = True,
        use_activation: bool = True,
    ):
        super().__init__()
        self.use_bn = use_bn
        self.use_activation = use_activation
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride,
            padding=0, output_padding=0, bias=False
        )
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        if use_activation:
            self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        x = self.conv(x)

        # Crop or pad to match target size
        if target_size is not None:
            h, w = x.shape[2], x.shape[3]
            th, tw = target_size

            if h > th:
                x = x[:, :, :th, :]
            elif h < th:
                x = F.pad(x, (0, 0, 0, th - h))

            if w > tw:
                x = x[:, :, :, :tw]
            elif w < tw:
                x = F.pad(x, (0, tw - w, 0, 0))

        if self.use_bn:
            x = self.bn(x)
        if self.use_activation:
            x = self.activation(x)
        return x


class InferenceGenerator(nn.Module):
    """Inference generator G_I: 2D U-Net architecture.

    Input: Lead-I (B, 1, 16, 512).
    Output: 12-lead ECG (B, 1, 16, 512) and latent z_i.
    """

    def __init__(
        self,
        encoder_filters: List[int] = [64, 128, 256, 512, 1024],
        decoder_filters: List[int] = [512, 256, 128, 64, 1],
        kernel_size: Tuple[int, int] = (2, 4),
        encoder_strides: List[List[int]] = None,
        decoder_strides: List[List[int]] = None,
    ):
        super().__init__()

        if encoder_strides is None:
            encoder_strides = [[2, 2], [2, 2], [2, 2], [2, 2], [1, 2]]
        if decoder_strides is None:
            decoder_strides = [[1, 2], [2, 2], [2, 2], [2, 2], [2, 2]]

        self.encoder_strides = encoder_strides
        self.decoder_strides = decoder_strides
        self.kernel_size = kernel_size

        # Encoder
        self.encoders = nn.ModuleList()
        in_ch = 1
        for i, out_ch in enumerate(encoder_filters):
            use_bn = (i > 0)
            self.encoders.append(
                EncoderBlock2D(
                    in_ch, out_ch, kernel_size,
                    tuple(encoder_strides[i]), use_bn
                )
            )
            in_ch = out_ch

        # Decoder
        self.decoders = nn.ModuleList()
        in_ch = encoder_filters[-1]
        for i, out_ch in enumerate(decoder_filters):
            if i > 0:
                in_ch = decoder_filters[i - 1] + encoder_filters[-(i + 1)]
            use_bn = (i < len(decoder_filters) - 1)
            use_activation = (i < len(decoder_filters) - 1)
            self.decoders.append(
                DecoderBlock2D(
                    in_ch, out_ch, kernel_size,
                    tuple(decoder_strides[i]), use_bn, use_activation
                )
            )
            in_ch = out_ch

        self.encoder_filters = encoder_filters
        self.decoder_filters = decoder_filters

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor, shape (B, 1, 16, 512).

        Returns:
            output: Generated 12-lead ECG, shape (B, 1, 16, 512).
            latent: Latent variable, shape (B, C, H, W).
        """
        target_size = x.shape[2:]

        # Encode with recorded sizes for skip connections
        encoder_outputs = []
        encoder_sizes = []
        h = x
        for encoder in self.encoders:
            encoder_sizes.append(h.shape[2:])
            h = encoder(h)
            encoder_outputs.append(h)

        latent = h

        # Decode with skip connections
        for i, decoder in enumerate(self.decoders):
            if i == 0:
                tgt_size = encoder_sizes[-(i + 1)] if i < len(encoder_sizes) else None
                h = decoder(h, target_size=tgt_size)
            else:
                skip = encoder_outputs[-(i + 1)]
                if h.shape[2:] != skip.shape[2:]:
                    h = F.interpolate(h, size=skip.shape[2:], mode='bilinear', align_corners=False)
                h = torch.cat([h, skip], dim=1)

                if i < len(encoder_sizes) - 1:
                    tgt_size = encoder_sizes[-(i + 2)]
                else:
                    tgt_size = target_size
                h = decoder(h, target_size=tgt_size)

        if h.shape[2:] != target_size:
            h = F.interpolate(h, size=target_size, mode='bilinear', align_corners=False)

        return h, latent


class LabelGenerator(nn.Module):
    """Label generator G_L: autoencoder (no skip connections)."""

    def __init__(
        self,
        encoder_filters: List[int] = [64, 128, 256, 512, 1024],
        decoder_filters: List[int] = [512, 256, 128, 64, 1],
        kernel_size: Tuple[int, int] = (2, 4),
        encoder_strides: List[List[int]] = None,
        decoder_strides: List[List[int]] = None,
    ):
        super().__init__()

        if encoder_strides is None:
            encoder_strides = [[2, 2], [2, 2], [2, 2], [2, 2], [1, 2]]
        if decoder_strides is None:
            decoder_strides = [[1, 2], [2, 2], [2, 2], [2, 2], [2, 2]]

        self.encoder_strides = encoder_strides
        self.decoder_strides = decoder_strides
        self.kernel_size = kernel_size

        # Encoder
        self.encoders = nn.ModuleList()
        in_ch = 1
        for i, out_ch in enumerate(encoder_filters):
            use_bn = (i > 0)
            self.encoders.append(
                EncoderBlock2D(
                    in_ch, out_ch, kernel_size,
                    tuple(encoder_strides[i]), use_bn
                )
            )
            in_ch = out_ch

        # Decoder (no skip connections)
        self.decoders = nn.ModuleList()
        in_ch = encoder_filters[-1]
        for i, out_ch in enumerate(decoder_filters):
            use_bn = (i < len(decoder_filters) - 1)
            use_activation = (i < len(decoder_filters) - 1)
            self.decoders.append(
                DecoderBlock2D(
                    in_ch, out_ch, kernel_size,
                    tuple(decoder_strides[i]), use_bn, use_activation
                )
            )
            in_ch = out_ch

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor, shape (B, 1, 16, 512).

        Returns:
            output: Reconstructed Lead-I, shape (B, 1, 16, 512).
            latent: Latent variable, shape (B, C, H, W).
        """
        target_size = x.shape[2:]

        encoder_sizes = []
        h = x
        for encoder in self.encoders:
            encoder_sizes.append(h.shape[2:])
            h = encoder(h)

        latent = h

        for i, decoder in enumerate(self.decoders):
            if i < len(encoder_sizes) - 1:
                tgt_size = encoder_sizes[-(i + 2)]
            else:
                tgt_size = target_size
            h = decoder(h, target_size=tgt_size)

        if h.shape[2:] != target_size:
            h = F.interpolate(h, size=target_size, mode='bilinear', align_corners=False)

        return h, latent


class Discriminator1D(nn.Module):
    """Discriminator D: 1D U-Net structure."""

    def __init__(
        self,
        encoder_filters: List[int] = [32, 64, 128, 256, 512],
        decoder_filters: List[int] = [256, 128, 64, 32, 1],
        kernel_sizes: List[int] = [64, 32, 16, 8, 4],
        strides: List[int] = [4, 4, 4, 2, 2],
    ):
        super().__init__()

        self.kernel_sizes = kernel_sizes
        self.strides = strides

        # Encoder
        self.encoders = nn.ModuleList()
        in_ch = 2  # Concatenated input and target
        for i, out_ch in enumerate(encoder_filters):
            use_bn = (i > 0)
            padding = kernel_sizes[i] // 2
            self.encoders.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_ch, out_ch, (1, kernel_sizes[i]), (1, strides[i]),
                        padding=(0, padding), bias=False
                    ),
                    nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_ch = out_ch

        # Decoder
        self.decoders = nn.ModuleList()
        in_ch = encoder_filters[-1]
        for i, out_ch in enumerate(decoder_filters):
            if i > 0:
                in_ch = decoder_filters[i - 1] + encoder_filters[-(i + 1)]
            use_bn = (i < len(decoder_filters) - 1)
            use_activation = (i < len(decoder_filters) - 1)
            padding = kernel_sizes[-(i + 1)] // 2

            layers = [
                nn.ConvTranspose2d(
                    in_ch, out_ch, (1, kernel_sizes[-(i + 1)]), (1, strides[-(i + 1)]),
                    padding=(0, padding), bias=False
                )
            ]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            if use_activation:
                layers.append(nn.ReLU(inplace=True))

            self.decoders.append(nn.Sequential(*layers))
            in_ch = out_ch

        self.encoder_filters = encoder_filters
        self.decoder_filters = decoder_filters

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input Lead-I, shape (B, 1, 16, 512).
            target: Real or generated 12-lead ECG, shape (B, 1, 16, 512).

        Returns:
            Discrimination result.
        """
        target_size = x.shape[2:]
        h = torch.cat([x, target], dim=1)

        # Encode
        encoder_outputs = []
        for encoder in self.encoders:
            h = encoder(h)
            encoder_outputs.append(h)

        # Decode
        for i, decoder in enumerate(self.decoders):
            if i == 0:
                h = decoder(h)
            else:
                skip = encoder_outputs[-(i + 1)]
                if h.shape[2:] != skip.shape[2:]:
                    h = F.interpolate(h, size=skip.shape[2:], mode='bilinear', align_corners=False)
                h = torch.cat([h, skip], dim=1)
                h = decoder(h)

        if h.shape[2:] != target_size:
            h = F.interpolate(h, size=target_size, mode='bilinear', align_corners=False)

        return torch.sigmoid(h)


class EKGAN(nn.Module):
    """EKGAN: Conditional GAN for 12-lead ECG reconstruction.

    Components:
        1. Inference generator G_I: generates 12-lead from Lead-I.
        2. Label generator G_L: autoencoder for latent alignment.
        3. Discriminator D: distinguishes real vs. generated ECG.
    """

    def __init__(self, config: dict):
        super().__init__()

        model_config = config.get('model', {})
        data_config = config.get('data', {})

        self.signal_length = data_config.get('signal_length', 512)
        self.num_leads = data_config.get('num_leads', 12)
        self.input_height = model_config.get('input_height', 16)
        self.zero_padding_rows = model_config.get('zero_padding_rows', 2)

        # Inference generator
        ig_config = model_config.get('inference_generator', {})
        self.inference_generator = InferenceGenerator(
            encoder_filters=ig_config.get('encoder_filters', [64, 128, 256, 512, 1024]),
            decoder_filters=ig_config.get('decoder_filters', [512, 256, 128, 64, 1]),
            kernel_size=tuple(ig_config.get('kernel_size', [2, 4])),
            encoder_strides=ig_config.get('encoder_strides'),
            decoder_strides=ig_config.get('decoder_strides'),
        )

        # Label generator
        lg_config = model_config.get('label_generator', {})
        self.label_generator = LabelGenerator(
            encoder_filters=lg_config.get('encoder_filters', [64, 128, 256, 512, 1024]),
            decoder_filters=lg_config.get('decoder_filters', [512, 256, 128, 64, 1]),
            kernel_size=tuple(lg_config.get('kernel_size', [2, 4])),
            encoder_strides=lg_config.get('encoder_strides'),
            decoder_strides=lg_config.get('decoder_strides'),
        )

        # Discriminator
        d_config = model_config.get('discriminator', {})
        self.discriminator = Discriminator1D(
            encoder_filters=d_config.get('encoder_filters', [32, 64, 128, 256, 512]),
            decoder_filters=d_config.get('decoder_filters', [256, 128, 64, 32, 1]),
            kernel_sizes=d_config.get('kernel_sizes', [64, 32, 16, 8, 4]),
            strides=d_config.get('strides', [4, 4, 4, 2, 2]),
        )

        # Weight initialization
        init_config = config.get('training', {}).get('init', {})
        self._init_weights(
            init_config.get('mean', 0.0),
            init_config.get('std', 0.02)
        )

    def _init_weights(self, mean: float = 0.0, std: float = 0.02):
        """Initialize all sub-network weights."""
        self.inference_generator.apply(lambda m: weights_init_normal(m, mean, std))
        self.label_generator.apply(lambda m: weights_init_normal(m, mean, std))
        self.discriminator.apply(lambda m: weights_init_normal(m, mean, std))

    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Prepare input: replicate Lead-I 12 times and add zero-padding rows.

        Args:
            x: Lead-I, shape (B, 1, L).

        Returns:
            Padded input, shape (B, 1, 16, L).
        """
        batch_size = x.shape[0]
        length = x.shape[-1]

        x_expanded = x.repeat(1, self.num_leads, 1)  # (B, 12, L)

        padding = torch.zeros(batch_size, self.zero_padding_rows * 2, length, device=x.device)
        x_padded = torch.cat([
            padding[:, :self.zero_padding_rows, :],
            x_expanded,
            padding[:, self.zero_padding_rows:, :]
        ], dim=1)  # (B, 16, L)

        x_padded = x_padded.unsqueeze(1)  # (B, 1, 16, L)
        return x_padded

    def prepare_target(self, x: torch.Tensor) -> torch.Tensor:
        """Prepare target: add zero-padding rows.

        Args:
            x: 12-lead ECG, shape (B, 12, L).

        Returns:
            Padded target, shape (B, 1, 16, L).
        """
        batch_size = x.shape[0]
        length = x.shape[-1]

        padding = torch.zeros(batch_size, self.zero_padding_rows * 2, length, device=x.device)
        x_padded = torch.cat([
            padding[:, :self.zero_padding_rows, :],
            x,
            padding[:, self.zero_padding_rows:, :]
        ], dim=1)  # (B, 16, L)

        x_padded = x_padded.unsqueeze(1)  # (B, 1, 16, L)
        return x_padded

    def extract_output(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 12-lead output by removing zero-padding.

        Args:
            x: Network output, shape (B, 1, 16, L).

        Returns:
            12-lead ECG, shape (B, 12, L).
        """
        x = x.squeeze(1)  # (B, 16, L)
        x = x[:, self.zero_padding_rows:-self.zero_padding_rows, :]  # (B, 12, L)
        return x

    def forward(
        self,
        lead_i: torch.Tensor,
        return_latent: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Inference-only forward pass.

        Args:
            lead_i: Lead-I input, shape (B, 1, L).
            return_latent: Whether to return latent variables.

        Returns:
            Dictionary with generated results.
        """
        x = self.prepare_input(lead_i)
        output, latent = self.inference_generator(x)
        ecg_12lead = self.extract_output(output)

        result = {'output': ecg_12lead}
        if return_latent:
            result['latent'] = latent
        return result

    def train_step(
        self,
        lead_i: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Training step: returns all intermediate results for loss computation.

        Args:
            lead_i: Lead-I input, shape (B, 1, L).
            target: Target 12-lead ECG, shape (B, 12, L).

        Returns:
            Dictionary with all intermediate results.
        """
        x = self.prepare_input(lead_i)
        target_padded = self.prepare_target(target)

        ig_output, ig_latent = self.inference_generator(x)
        lg_output, lg_latent = self.label_generator(x)

        disc_real = self.discriminator(x, target_padded)
        disc_fake = self.discriminator(x, ig_output)

        return {
            'ig_output': ig_output,
            'ig_latent': ig_latent,
            'lg_output': lg_output,
            'lg_latent': lg_latent,
            'disc_real': disc_real,
            'disc_fake': disc_fake,
            'target_padded': target_padded,
            'input_padded': x,
        }


def build_ekgan(config: dict) -> EKGAN:
    """Build EKGAN model from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        EKGAN model instance.
    """
    return EKGAN(config)
