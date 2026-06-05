"""
Lightweight HiFi-GAN inference model for neural pitch correction.

Loads weights trained by scripts/train.py and performs inference
(no discriminator, no training-only components).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Minimal HiFi-GAN generator (same architecture as scripts/hifi_gan.py)
# ---------------------------------------------------------------------------

def _get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class _ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.convs = nn.ModuleList()
        for d in dilations:
            self.convs.append(nn.Conv1d(
                channels, channels, kernel_size,
                dilation=d, padding=_get_padding(kernel_size, d),
            ))

    def forward(self, x):
        for conv in self.convs:
            residual = x
            x = F.leaky_relu(x, 0.1)
            x = conv(x)
            x = x + residual
        return x


class _MRF(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.resblocks = nn.ModuleList([
            _ResBlock(channels, 3, [1, 3, 5]),
            _ResBlock(channels, 7, [1, 3, 5]),
            _ResBlock(channels, 11, [1, 3, 5]),
        ])

    def forward(self, x):
        return sum(block(x) for block in self.resblocks) / len(self.resblocks)



def _make_pitch_feature(pitch_hz: torch.Tensor, n_frames: int) -> torch.Tensor:
    """Convert Hz pitch to [-1, 1] feature map."""
    B, T = pitch_hz.shape
    if T != n_frames:
        pitch_hz = F.interpolate(
            pitch_hz.unsqueeze(1), size=n_frames,
            mode="linear", align_corners=False).squeeze(1)
    pitch_feat = torch.zeros(B, 1, n_frames, device=pitch_hz.device)
    voiced = pitch_hz > 20.0
    if voiced.any():
        log_pitch = torch.log2(torch.clamp(pitch_hz, min=20.0) / 440.0)
        pitch_feat[:, 0, :] = torch.tanh(log_pitch / 4.0)
    return pitch_feat


class InferenceGenerator(nn.Module):
    """HiFi-GAN generator for inference only.

    Architecture is identical to the training HiFiGANGenerator (hifi_gan.py)
    but WITHOUT weight_norm — the training script calls remove_weight_norm()
    before export, so this model loads the exported weights directly.

    See hifi_gan.py for the training-time twin; keep the two in sync.
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
        self.upsample_rates = upsample_rates

        # --- Input projection ---
        if hubert_dim > 0:
            self.hubert_proj = nn.Conv1d(hubert_dim, h_channels // 2, kernel_size=1)
            self.input_channels = h_channels // 2 + pitch_bins
        else:
            self.input_channels = mel_bins + pitch_bins

        # --- Pre-convolution (no weight_norm — matches exported weights) ---
        self.conv_pre = nn.Conv1d(
            self.input_channels, h_channels, kernel_size=7,
            stride=1, padding=3,
        )

        # --- Upsampling blocks (same topology as HiFiGANGenerator) ---
        self.upsamples = nn.ModuleList()
        self.mrfs = nn.ModuleList()

        in_ch = h_channels
        for i, (rate, kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            out_ch = upsample_initial_channel // (2 ** (i + 1))
            self.upsamples.append(
                nn.ConvTranspose1d(
                    in_ch, out_ch, kernel,
                    stride=rate,
                    padding=(kernel - rate) // 2,
                )
            )
            self.mrfs.append(_MRF(out_ch))
            in_ch = out_ch

        # --- Post-convolution: hidden → 1 channel waveform ---
        self.conv_post = nn.Conv1d(in_ch, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, input_feat: torch.Tensor, pitch: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            input_feat: (B, hubert_dim, T) HuBERT features, or
                        (B, mel_bins, T)  mel spectrogram (legacy mode)
            pitch:      (B, T_pitch) pitch contour in Hz, or None

        Returns:
            wav: (B, 1, T_audio) generated waveform
        """
        n_frames = input_feat.shape[2]

        # --- Build input: content features + pitch conditioning ---
        if self.hubert_dim > 0:
            x = F.leaky_relu(self.hubert_proj(input_feat), 0.1)
            if pitch is not None:
                pitch_feat = _make_pitch_feature(pitch, n_frames)
                x = torch.cat([x, pitch_feat], dim=1)
            else:
                x = torch.cat([
                    x,
                    torch.zeros(x.shape[0], 1, n_frames, device=x.device),
                ], dim=1)
        else:
            if pitch is not None:
                pitch_feat = _make_pitch_feature(pitch, n_frames)
                x = torch.cat([input_feat, pitch_feat], dim=1)
            else:
                x = torch.cat([
                    input_feat,
                    torch.zeros(input_feat.shape[0], 1, n_frames, device=input_feat.device),
                ], dim=1)

        # --- Generator forward ---
        x = self.conv_pre(x)

        for upsample, mrf in zip(self.upsamples, self.mrfs):
            x = F.leaky_relu(x, 0.1)
            x = upsample(x)
            x = mrf(x)

        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        return torch.tanh(x)


# ---------------------------------------------------------------------------
# HuBERT feature extraction for inference
# ---------------------------------------------------------------------------

_HUBERT_CACHE: dict = {}


def _get_hubert_model(device="cpu"):
    """Load HuBERT model lazily."""
    key = f"hubert::{device}"
    if key not in _HUBERT_CACHE:
        from transformers import HubertModel, Wav2Vec2FeatureExtractor
        _HUBERT_CACHE["extractor"] = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/hubert-base-ls960")
        _HUBERT_CACHE[key] = HubertModel.from_pretrained(
            "facebook/hubert-base-ls960").to(device).eval()
    return _HUBERT_CACHE["extractor"], _HUBERT_CACHE[key]


def extract_hubert_features(y: torch.Tensor, sr: int) -> torch.Tensor:
    """Extract HuBERT features from audio waveform.

    Args:
        y: (T,) audio waveform (float32)
        sr: sample rate

    Returns:
        (768, T_hubert) HuBERT features
    """
    y_np = y.cpu().numpy()
    extractor, model = _get_hubert_model(y.device.type)
    if sr != 16000:
        import librosa
        y_np = librosa.resample(y_np, orig_sr=sr, target_sr=16000)
    inputs = extractor(y_np, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**{k: v.to(y.device) for k, v in inputs.items()})
    feats = outputs.last_hidden_state.squeeze(0)  # (T, 768)
    return feats.T.contiguous()  # (768, T)

# Model loader
# ---------------------------------------------------------------------------

_model_cache: dict = {}


def load_model(model_path: str, device: str = "cpu") -> InferenceGenerator:
    """Load a trained tuner model. Auto-detects HuBERT vs mel mode."""
    if model_path in _model_cache:
        return _model_cache[model_path]

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    has_hubert = any("hubert_proj" in k for k in state_dict)
    if has_hubert:
        model = InferenceGenerator(hubert_dim=768)
    else:
        model = InferenceGenerator()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    _model_cache[model_path] = model
    return model


# ---------------------------------------------------------------------------
# Feature extraction for inference
# ---------------------------------------------------------------------------

def _mel_basis(n_fft: int, sr: int, n_mels: int) -> torch.Tensor:
    from librosa.filters import mel as mel_fn
    return torch.from_numpy(mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels)).float()


def extract_mel_torch(y: torch.Tensor, sr: int, n_fft: int = 1024,
                      hop: int = 256, win: int = 1024,
                      n_mels: int = 80) -> torch.Tensor:
    """Extract log-mel spectrogram. y shape: (B, T) or (T,)."""
    if y.ndim == 1:
        y = y.unsqueeze(0)
    window = torch.hann_window(win, device=y.device)
    spec = torch.stft(y, n_fft, hop, win, window, return_complex=True).abs() ** 2
    mel_b = _mel_basis(n_fft, sr, n_mels).to(y.device)
    mel = torch.matmul(mel_b, spec)
    return torch.log(torch.clamp(mel, min=1e-5))
