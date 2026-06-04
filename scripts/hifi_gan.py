"""
HiFi-GAN neural vocoder with pitch conditioning.

Based on "HiFi-GAN: Generative Adversarial Networks for Efficient
and High Fidelity Speech Synthesis" (Kong et al., NeurIPS 2020).

Extended with pitch conditioning for pitch correction task:
- Input: mel spectrogram (80 bins) + pitch feature (1 channel) = 81 channels
- Output: waveform

Architecture:
    Generator: mel+pitcch → Conv1d → Upsample×4 → MRF blocks → Conv1d → waveform
    MPD: Multi-Period Discriminator
    MSD: Multi-Scale Discriminator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# ---------------------------------------------------------------------------
# Helper: weight normalization wrapper
# ---------------------------------------------------------------------------

def init_weights(m, mean=0.0, std=0.01):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


# ---------------------------------------------------------------------------
# Residual block inside MRF
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.convs = nn.ModuleList()
        for d in dilations:
            padding = get_padding(kernel_size, d)
            self.convs.append(
                weight_norm(nn.Conv1d(
                    channels, channels, kernel_size,
                    dilation=d, padding=padding
                ))
            )

    def forward(self, x):
        for conv in self.convs:
            residual = x
            x = F.leaky_relu(x, 0.1)
            x = conv(x)
            x = x + residual
        return x


# ---------------------------------------------------------------------------
# Multi-Receptive Field Fusion block
# ---------------------------------------------------------------------------

class MRF(nn.Module):
    """Multi-Receptive Field Fusion: parallel residual blocks with different
    kernel sizes and dilation rates, summed together."""
    def __init__(self, channels):
        super().__init__()
        self.resblocks = nn.ModuleList([
            ResBlock(channels, kernel_size=3, dilations=[1, 3, 5]),
            ResBlock(channels, kernel_size=7, dilations=[1, 3, 5]),
            ResBlock(channels, kernel_size=11, dilations=[1, 3, 5]),
        ])

    def forward(self, x):
        out = sum(block(x) for block in self.resblocks)
        return out / len(self.resblocks)


# ---------------------------------------------------------------------------
# HiFi-GAN Generator (with pitch conditioning)
# ---------------------------------------------------------------------------

class HiFiGANGenerator(nn.Module):
    """HiFi-GAN generator with optional pitch and HuBERT conditioning.

    Supports two input modes:
    1. mel + pitch (legacy): input channels = mel_bins + pitch_bins
    2. hubert + pitch (new): hubert features (768d) projected to h_channels//2,
       then concat with pitch => total = h_channels//2 + pitch_bins

    Args:
        mel_bins: number of mel frequency bins (default 80, used when hubert_dim=0)
        pitch_bins: number of pitch feature channels (default 1, set 0 to disable)
        hubert_dim: HuBERT feature dimension (default 768, set 0 to use mel input)
        h_channels: hidden channels after pre-conv (default 512)
        upsample_rates: upsampling factors per block
        upsample_kernel_sizes: kernel sizes for upsampling convolutions
        upsample_initial_channel: hidden channels before first upsampling
    """

    def __init__(
        self,
        mel_bins: int = 80,
        pitch_bins: int = 1,
        hubert_dim: int = 0,
        h_channels: int = 512,
        upsample_rates: tuple = (8, 8, 2, 2),
        upsample_kernel_sizes: tuple = (16, 16, 4, 4),
        upsample_initial_channel: int = 256,
    ):
        super().__init__()

        self.mel_bins = mel_bins
        self.pitch_bins = pitch_bins
        self.hubert_dim = hubert_dim

        if hubert_dim > 0:
            # HuBERT mode: project 768-dim HuBERT features → h_channels//2
            self.hubert_proj = nn.Conv1d(hubert_dim, h_channels // 2, kernel_size=1)
            self.input_channels = h_channels // 2 + pitch_bins
        else:
            # Legacy mode: direct mel input
            self.input_channels = mel_bins + pitch_bins

        # Pre-convolution: input → hidden
        self.conv_pre = weight_norm(nn.Conv1d(
            self.input_channels, h_channels, kernel_size=7,
            stride=1, padding=3
        ))

        # Upsampling blocks
        self.upsamples = nn.ModuleList()
        self.mrfs = nn.ModuleList()

        in_ch = h_channels
        for i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            out_ch = upsample_initial_channel // (2 ** (i + 1))
            self.upsamples.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        in_ch, out_ch, kernel,
                        stride=rate,
                        padding=(kernel - rate) // 2,
                    )
                )
            )
            self.mrfs.append(MRF(out_ch))
            in_ch = out_ch

        self.upsample_rates = upsample_rates

        # Post-convolution: hidden → 1 channel (waveform)
        self.conv_post = weight_norm(nn.Conv1d(in_ch, 1, kernel_size=7, stride=1, padding=3))

        self.apply(init_weights)

    def _make_pitch_feature(self, pitch_hz: torch.Tensor, n_frames: int) -> torch.Tensor:
        """Convert pitch contour to a feature map.

        Args:
            pitch_hz: (B, T) pitch values in Hz, 0 where unvoiced
            n_frames: target frame count

        Returns:
            pitch_feat: (B, 1, n_frames) normalized pitch feature in [-1, 1]
        """
        B, T = pitch_hz.shape

        # Interpolate to target frame count
        if T != n_frames:
            pitch_hz = F.interpolate(
                pitch_hz.unsqueeze(1),
                size=n_frames,
                mode="linear",
                align_corners=False,
            ).squeeze(1)

        # Normalize: log2(hz/440) → tanh(x/4.0) → [-1, 1]
        pitch_feat = torch.zeros(B, 1, n_frames, device=pitch_hz.device)
        voiced = pitch_hz > 20.0
        if voiced.any():
            log_pitch = torch.log2(torch.clamp(pitch_hz, min=20.0) / 440.0)
            pitch_feat[:, 0, :] = torch.tanh(log_pitch / 4.0)

        return pitch_feat

    def forward(self, input_feat: torch.Tensor, pitch: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        In mel mode (hubert_dim=0):
            input_feat: (B, mel_bins, T) mel spectrogram
        In HuBERT mode (hubert_dim > 0):
            input_feat: (B, hubert_dim, T) HuBERT features

        Args:
            input_feat: content features (mel or HuBERT)
            pitch: (B, T_pitch) pitch contour in Hz, or None to skip conditioning

        Returns:
            wav: (B, 1, T_audio) generated waveform
        """
        n_frames = input_feat.shape[2]

        if self.hubert_dim > 0:
            # HuBERT mode: project then concat pitch
            x = F.leaky_relu(self.hubert_proj(input_feat), 0.1)
            if pitch is not None:
                pitch_feat = self._make_pitch_feature(pitch, n_frames)
                x = torch.cat([x, pitch_feat], dim=1)
            else:
                x = torch.cat([x, torch.zeros(x.shape[0], 1, n_frames, device=x.device)], dim=1)
        else:
            # Legacy mel mode
            if pitch is not None:
                pitch_feat = self._make_pitch_feature(pitch, n_frames)
                x = torch.cat([input_feat, pitch_feat], dim=1)
            else:
                x = torch.cat([input_feat, torch.zeros(input_feat.shape[0], 1, n_frames, device=input_feat.device)], dim=1)

        x = self.conv_pre(x)

        for upsample, mrf in zip(self.upsamples, self.mrfs):
            x = F.leaky_relu(x, 0.1)
            x = upsample(x)
            x = mrf(x)

        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        for module in self.modules():
            if hasattr(module, "weight_g"):
                nn.utils.remove_weight_norm(module)


# ---------------------------------------------------------------------------
# Discriminators
# ---------------------------------------------------------------------------

class PeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator: operates on reshaped 2D views of the
    1D waveform at different periods."""

    def __init__(self, period: int):
        super().__init__()
        self.period = period

        # 2D convolutions
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (5, 1), stride=(3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(32, 128, (5, 1), stride=(3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(128, 512, (5, 1), stride=(3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(512, 1024, (5, 1), stride=(3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (5, 1), padding=(2, 0))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), padding=(1, 0)))

    def forward(self, x):
        # x: (B, 1, T)
        b, c, t_audio = x.shape

        # Pad to multiple of period
        if t_audio % self.period != 0:
            pad = self.period - (t_audio % self.period)
            x = F.pad(x, (0, pad), mode="reflect")
            t_audio = t_audio + pad

        # Reshape: (B, 1, T) → (B, 1, P, T/P)
        x = x.view(b, 1, t_audio // self.period, self.period)

        fmaps = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmaps.append(x)

        x = self.conv_post(x)
        fmaps.append(x)

        return x.flatten(1), fmaps


class ScaleDiscriminator(nn.Module):
    """Multi-Scale Discriminator: operates on waveforms at different
    resolutions via average pooling."""

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(1, 128, 15, stride=1, padding=7)),
            weight_norm(nn.Conv1d(128, 128, 41, stride=2, groups=4, padding=20)),
            weight_norm(nn.Conv1d(128, 256, 41, stride=2, groups=16, padding=20)),
            weight_norm(nn.Conv1d(256, 512, 41, stride=4, groups=16, padding=20)),
            weight_norm(nn.Conv1d(512, 1024, 41, stride=4, groups=16, padding=20)),
            weight_norm(nn.Conv1d(1024, 1024, 41, stride=1, groups=16, padding=20)),
            weight_norm(nn.Conv1d(1024, 1024, 5, stride=1, padding=2)),
        ])
        self.conv_post = weight_norm(nn.Conv1d(1024, 1, 3, stride=1, padding=1))

    def forward(self, x):
        fmaps = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmaps.append(x)
        x = self.conv_post(x)
        fmaps.append(x)
        return x.flatten(1), fmaps


class HiFiGANDiscriminator(nn.Module):
    """Combined MPD + MSD discriminator."""

    def __init__(self, mpd_periods: list[int] | None = None):
        super().__init__()
        if mpd_periods is None:
            mpd_periods = [2, 3, 5, 7, 11]
        self.mpd = nn.ModuleList([PeriodDiscriminator(p) for p in mpd_periods])
        self.msd = nn.ModuleList([ScaleDiscriminator() for _ in range(3)])

    def forward(self, x):
        mpd_results = []
        msd_results = []

        # MPD
        for disc in self.mpd:
            score, fmaps = disc(x)
            mpd_results.append((score, fmaps))

        # MSD
        for i, disc in enumerate(self.msd):
            if i > 0:
                x_in = F.avg_pool1d(x, kernel_size=4, stride=2, padding=2)
            else:
                x_in = x
            score, fmaps = disc(x_in)
            msd_results.append((score, fmaps))

        return mpd_results, msd_results


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class MelSpectrogramLoss(nn.Module):
    """Multi-resolution mel-spectrogram loss."""

    def __init__(
        self,
        sample_rate: int = 22050,
        n_ffts: tuple = (1024, 2048, 512),
        hop_lengths: tuple = (256, 512, 128),
        win_lengths: tuple = (1024, 2048, 512),
        n_mels: int = 80,
    ):
        super().__init__()
        self.loss_fn = nn.L1Loss()
        self.n_ffts = n_ffts
        self.hop_lengths = hop_lengths
        self.win_lengths = win_lengths
        self.n_mels = n_mels

        # Mel filter banks
        self.mel_basis = {}
        for n_fft in n_ffts:
            self.mel_basis[n_fft] = self._mel_basis(n_fft, sample_rate)

    def _mel_basis(self, n_fft, sr):
        from librosa.filters import mel
        return torch.from_numpy(mel(sr=sr, n_fft=n_fft, n_mels=self.n_mels)).float()

    def _mel_spectrogram(self, y, n_fft, hop, win):
        device = y.device
        mel_b = self.mel_basis[n_fft].to(device)
        window = torch.hann_window(win, device=device)

        # STFT
        y = y.squeeze(1)  # (B, T)
        spec = torch.stft(
            y, n_fft, hop, win, window,
            return_complex=True
        ).abs()
        spec = spec ** 2  # power spectrogram

        # Mel
        mel_spec = torch.matmul(mel_b, spec)
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        return mel_spec

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for n_fft, hop, win in zip(self.n_ffts, self.hop_lengths, self.win_lengths):
            mel_pred = self._mel_spectrogram(y_pred, n_fft, hop, win)
            mel_true = self._mel_spectrogram(y_true, n_fft, hop, win)
            # Align time dimension: generator upsampling may produce +/- 1 frame
            min_t = min(mel_pred.shape[2], mel_true.shape[2])
            mel_pred = mel_pred[:, :, :min_t]
            mel_true = mel_true[:, :, :min_t]
            loss += self.loss_fn(mel_pred, mel_true)
        return loss / len(self.n_ffts)


def feature_loss(fmap_real: list, fmap_fake: list) -> torch.Tensor:
    """Feature matching loss between discriminator feature maps."""
    loss = 0.0
    for r, f in zip(fmap_real, fmap_fake):
        for rl, fl in zip(r, f):
            # Align spatial dims: generator output may differ by a few samples
            min_dim = min(rl.shape[-1], fl.shape[-1])
            if rl.ndim >= 3:
                rl = rl[..., :min_dim]
                fl = fl[..., :min_dim]
            loss += F.l1_loss(fl, rl.detach())
    return loss


def discriminator_loss(real_scores, fake_scores) -> torch.Tensor:
    """Hinge loss for discriminator."""
    loss_real = sum(torch.mean(F.relu(1 - s)) for s in real_scores)
    loss_fake = sum(torch.mean(F.relu(1 + s)) for s in fake_scores)
    return loss_real + loss_fake


def generator_loss(fake_scores) -> torch.Tensor:
    """Hinge loss for generator (adversarial part)."""
    return sum(-torch.mean(s) for s in fake_scores)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def create_generator(
    mel_bins: int = 80,
    pitch_bins: int = 1,
    hubert_dim: int = 0,
    h_channels: int = 512,
    upsample_rates: tuple = (8, 8, 2, 2),
) -> HiFiGANGenerator:
    """Create a HiFi-GAN generator.

    Default config generates 22kHz audio from ~86 Hz frame rate.
    86 * 8 * 8 * 2 * 2 = 22050 Hz.

    When hubert_dim > 0 (e.g. 768), HuBERT projection layer is added
    and mel_bins is ignored.
    """
    return HiFiGANGenerator(
        mel_bins=mel_bins,
        pitch_bins=pitch_bins,
        hubert_dim=hubert_dim,
        h_channels=h_channels,
        upsample_rates=upsample_rates,
        upsample_kernel_sizes=(16, 16, 4, 4),
    )


def create_discriminator() -> HiFiGANDiscriminator:
    return HiFiGANDiscriminator()
