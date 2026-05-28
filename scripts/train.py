"""
Neural pitch correction model training.

Fine-tunes a HiFi-GAN vocoder with pitch conditioning on paired
(out_of_tune, clean) audio data. The model learns to reconstruct
clean vocals from out-of-tune input, using target pitch as a condition.

Usage:
    # Step 1: Generate training data
    python generate_training_data.py --input_dir data/clean/ --output_dir data/training/

    # Step 2: Train
    python train.py --data_dir data/training/ --checkpoint_dir checkpoints/

    # Step 3: The trained model weights → models/tuner.pth (for inference)
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from hifi_gan import (
    create_generator,
    create_discriminator,
    MelSpectrogramLoss,
    feature_loss,
    discriminator_loss,
    generator_loss,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TrainConfig:
    # Audio
    sample_rate: int = 22050
    hop_length: int = 256
    segment_ms: int = 800  # training segment duration in ms

    # Mel spectrogram
    mel_bins: int = 80
    n_fft: int = 1024
    win_length: int = 1024

    # Model
    h_channels: int = 512
    pitch_bins: int = 1

    # Training
    batch_size: int = 16
    learning_rate: float = 2e-4
    lr_decay: float = 0.999
    adam_b1: float = 0.8
    adam_b2: float = 0.99
    num_epochs: int = 100
    grad_clip: float = 5.0

    # Loss weights
    lambda_mel: float = 45.0
    lambda_fm: float = 2.0
    lambda_adv: float = 1.0

    # Logging
    log_interval: int = 100
    save_interval: int = 5  # epochs
    valid_interval: int = 5  # epochs

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PairedVocalDataset(Dataset):
    """Dataset of (out_of_tune, clean) vocal pairs."""

    def __init__(self, data_dir: str, segment_ms: int = 800, sample_rate: int = 22050):
        self.data_dir = Path(data_dir)
        self.segment_samples = int(segment_ms / 1000.0 * sample_rate)
        self.sample_rate = sample_rate

        with open(self.data_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        self.pairs = self.metadata["pairs"]
        self.pairs_dir = self.data_dir / "pairs"

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        import soundfile as sf

        pair = self.pairs[idx]
        clean_path = self.pairs_dir / pair["clean"]
        shifted_path = self.pairs_dir / pair["shifted"]

        y_clean, _ = sf.read(str(clean_path))
        y_shifted, _ = sf.read(str(shifted_path))

        # Load precomputed WORLD F0 (stored as float32 .npy)
        f0_path = self.pairs_dir / pair["f0"]
        f0_full = np.load(str(f0_path))

        # Ensure same length
        min_len = min(len(y_clean), len(y_shifted))
        y_clean = y_clean[:min_len]
        y_shifted = y_shifted[:min_len]

        # Random segment or pad (keep F0 aligned with audio)
        if min_len >= self.segment_samples:
            start = random.randint(0, min_len - self.segment_samples)
            y_clean = y_clean[start:start + self.segment_samples]
            y_shifted = y_shifted[start:start + self.segment_samples]

            # Slice F0: convert sample index to frame index (5ms hop = sample_rate * 0.005)
            frame_period = int(self.sample_rate * 0.005)
            f0_start = start // frame_period
            f0_end = (start + self.segment_samples + frame_period - 1) // frame_period
            f0_full = f0_full[f0_start:f0_end]
            expected_f0_frames = (self.segment_samples + frame_period - 1) // frame_period
            if f0_full.shape[0] < expected_f0_frames:
                f0_full = np.pad(f0_full, (0, expected_f0_frames - f0_full.shape[0]))
            else:
                f0_full = f0_full[:expected_f0_frames]
        else:
            y_clean = np.pad(y_clean, (0, self.segment_samples - min_len))
            y_shifted = np.pad(y_shifted, (0, self.segment_samples - min_len))

        return (
            torch.from_numpy(y_shifted).float(),
            torch.from_numpy(y_clean).float(),
            torch.from_numpy(f0_full).float(),
        )


# ---------------------------------------------------------------------------
# Mel & pitch extraction
# ---------------------------------------------------------------------------

# Cached mel filter banks: keyed by (sr, n_fft, n_mels, device)
_mel_basis_cache: dict = {}

def extract_mel(y: torch.Tensor, sr: int, n_fft: int, hop: int, win: int,
                n_mels: int) -> torch.Tensor:
    """Extract log-mel spectrogram from waveform."""
    window = torch.hann_window(win, device=y.device)
    spec = torch.stft(y, n_fft, hop, win, window, return_complex=True).abs()
    spec = spec ** 2  # power spectrum

    cache_key = (sr, n_fft, n_mels, y.device)
    if cache_key not in _mel_basis_cache:
        from librosa.filters import mel as mel_fn
        _mel_basis_cache[cache_key] = torch.from_numpy(
            mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels)
        ).float().to(y.device)
    mel_basis = _mel_basis_cache[cache_key]

    mel = torch.matmul(mel_basis, spec)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    return mel


def extract_pitch(y: np.ndarray, sr: int, hop_length: int) -> np.ndarray:
    """Extract pitch contour using pYIN (returns Hz, 0 for unvoiced)."""
    import librosa
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("E2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr, hop_length=hop_length,
    )
    f0 = np.nan_to_num(f0, nan=0.0)
    return f0.astype(np.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    torch.set_float32_matmul_precision("medium")
    config = TrainConfig()
    # Override from args
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    config.learning_rate = args.lr
    config.device = args.device or config.device

    device = torch.device(config.device)

    # Dataset & loader
    dataset = PairedVocalDataset(
        args.data_dir,
        segment_ms=config.segment_ms,
        sample_rate=config.sample_rate,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=(config.device == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    print(f"Dataset: {len(dataset)} pairs, {len(dataloader)} batches/epoch")

    # Models
    generator = create_generator(
        mel_bins=config.mel_bins,
        pitch_bins=config.pitch_bins,
        h_channels=config.h_channels,
    ).to(device)

    discriminator = create_discriminator().to(device)

    print(f"Generator params: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")

    # Load checkpoint if resuming
    start_epoch = 0
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    latest_ckpt = checkpoint_dir / "latest.pt"
    if args.resume and latest_ckpt.exists():
        ckpt = torch.load(latest_ckpt, map_location=device)
        generator.load_state_dict(ckpt["generator"])
        discriminator.load_state_dict(ckpt["discriminator"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    print("Compiling models (first step will be slow)...")
    generator = torch.compile(generator, mode="max-autotune")
    discriminator = torch.compile(discriminator, mode="max-autotune")

    # Optimizers
    opt_g = torch.optim.AdamW(
        generator.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_b1, config.adam_b2),
    )
    opt_d = torch.optim.AdamW(
        discriminator.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_b1, config.adam_b2),
    )

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, config.lr_decay)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, config.lr_decay)

    # Losses
    mel_loss_fn = MelSpectrogramLoss(
        sample_rate=config.sample_rate,
        n_mels=config.mel_bins,
    ).to(device)

    # Logger
    writer = SummaryWriter(checkpoint_dir / "logs")

    # Training loop
    global_step = 0
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, config.num_epochs):
        generator.train()
        discriminator.train()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0

        for batch in pbar:
            _profile = (global_step < 3 or global_step % config.log_interval == 0)
            if _profile:
                _t0 = time.perf_counter()

            y_shifted, y_clean, target_f0 = batch
            y_shifted = y_shifted.to(device)
            y_clean = y_clean.to(device)
            target_f0 = target_f0.to(device)
            # Align lengths: dataset may return slightly different durations
            min_len = min(y_shifted.shape[-1], y_clean.shape[-1])
            y_shifted = y_shifted[..., :min_len]
            y_clean = y_clean[..., :min_len]

            if _profile:
                torch.cuda.synchronize()
                _t_data = time.perf_counter() - _t0
                _t1 = time.perf_counter()

            # ----------------------------------------------------------------
            # Extract features (mel on GPU; F0 precomputed by WORLD)
            # ----------------------------------------------------------------
            with torch.no_grad():
                mel_shifted = extract_mel(
                    y_shifted, config.sample_rate, config.n_fft,
                    config.hop_length, config.win_length, config.mel_bins,
                )

                # Resample precomputed F0 to match mel frame count
                mel_frames = mel_shifted.shape[2]
                if target_f0.shape[-1] != mel_frames:
                    target_f0 = F.interpolate(
                        target_f0.unsqueeze(1),
                        size=mel_frames,
                        mode="linear",
                        align_corners=False,
                    ).squeeze(1)

            if _profile:
                torch.cuda.synchronize()
                _t_mel = time.perf_counter() - _t1
                _t2 = time.perf_counter()

            # ----------------------------------------------------------------
            # Train Discriminator
            # ----------------------------------------------------------------
            opt_d.zero_grad()

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    y_gen = generator(mel_shifted, target_f0)
                y_gen = y_gen[..., :y_clean.shape[-1]]

            mpd_real, msd_real = discriminator(y_clean.unsqueeze(1))
            mpd_fake, msd_fake = discriminator(y_gen.detach().float())

            loss_d = discriminator_loss(
                [r[0] for r in mpd_real] + [r[0] for r in msd_real],
                [f[0] for f in mpd_fake] + [f[0] for f in msd_fake],
            )
            scaler.scale(loss_d).backward()
            scaler.unscale_(opt_d)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), config.grad_clip)
            scaler.step(opt_d)

            if _profile:
                torch.cuda.synchronize()
                _t_dsc = time.perf_counter() - _t2
                _t3 = time.perf_counter()

            # ----------------------------------------------------------------
            # Train Generator
            # ----------------------------------------------------------------
            opt_g.zero_grad()

            with torch.cuda.amp.autocast():
                y_gen = generator(mel_shifted, target_f0)
            y_gen = y_gen[..., :y_clean.shape[-1]]

            mpd_fake, msd_fake = discriminator(y_gen.float())
            mpd_real, msd_real = discriminator(y_clean.unsqueeze(1))

            # Mel-spectrogram loss
            loss_mel = mel_loss_fn(y_gen, y_clean.unsqueeze(1))

            # Feature matching loss
            fm_real = [r[1] for r in mpd_real] + [r[1] for r in msd_real]
            fm_fake = [f[1] for f in mpd_fake] + [f[1] for f in msd_fake]
            loss_fm = feature_loss(fm_real, fm_fake)

            # Adversarial loss
            loss_adv = generator_loss(
                [f[0] for f in mpd_fake] + [f[0] for f in msd_fake]
            )

            loss_g = (
                config.lambda_mel * loss_mel +
                config.lambda_fm * loss_fm +
                config.lambda_adv * loss_adv
            )
            scaler.scale(loss_g).backward()
            scaler.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), config.grad_clip)
            scaler.step(opt_g)
            scaler.update()

            if _profile:
                torch.cuda.synchronize()
                _t_gen = time.perf_counter() - _t3

            # Logging
            epoch_loss_g += loss_g.item()
            epoch_loss_d += loss_d.item()

            if global_step % config.log_interval == 0:
                writer.add_scalar("train/loss_g", loss_g.item(), global_step)
                writer.add_scalar("train/loss_d", loss_d.item(), global_step)
                writer.add_scalar("train/loss_mel", loss_mel.item(), global_step)
                writer.add_scalar("train/loss_fm", loss_fm.item(), global_step)
                writer.add_scalar("train/loss_adv", loss_adv.item(), global_step)
                writer.add_scalar("train/lr", opt_g.param_groups[0]["lr"], global_step)

            if _profile:
                writer.add_scalar("profile/data_ms", _t_data * 1000, global_step)
                writer.add_scalar("profile/mel_ms", _t_mel * 1000, global_step)
                writer.add_scalar("profile/disc_ms", _t_dsc * 1000, global_step)
                writer.add_scalar("profile/gen_ms", _t_gen * 1000, global_step)
                pbar.set_postfix(
                    g=f"{loss_g.item():.2f}",
                    d=f"{loss_d.item():.2f}",
                    io=f"{_t_data*1000:.0f}ms",
                    ml=f"{_t_mel*1000:.0f}ms",
                    dc=f"{_t_dsc*1000:.0f}ms",
                    gn=f"{_t_gen*1000:.0f}ms",
                )
            else:
                pbar.set_postfix(
                    g=f"{loss_g.item():.2f}",
                    d=f"{loss_d.item():.2f}",
                    mel=f"{loss_mel.item():.1f}",
                )
            global_step += 1

        scheduler_g.step()
        scheduler_d.step()

        # Epoch summary
        avg_g = epoch_loss_g / len(dataloader)
        avg_d = epoch_loss_d / len(dataloader)
        print(f"  Epoch {epoch+1} avg loss — G: {avg_g:.2f}  D: {avg_d:.2f}")

        # Save checkpoint
        if (epoch + 1) % config.save_interval == 0:
            save_path = checkpoint_dir / f"epoch_{epoch+1:03d}.pt"
            torch.save({
                "epoch": epoch,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "opt_g": opt_g.state_dict(),
                "opt_d": opt_d.state_dict(),
                "config": {k: v for k, v in vars(config).items()
                           if not k.startswith("__")},
            }, save_path)
            print(f"  Saved: {save_path}")

        # Always save latest
        torch.save({
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
        }, latest_ckpt)

    # Final: export inference model
    generator.remove_weight_norm()
    export_path = Path(args.checkpoint_dir).parent / "models" / "tuner.pth"
    export_path.parent.mkdir(exist_ok=True)
    torch.save(generator.state_dict(), export_path)
    print(f"\nInference model exported to: {export_path}")

    writer.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train neural pitch corrector")

    # Data
    parser.add_argument("--data_dir", default="data/training/",
                        help="Directory with training pairs (from generate_training_data.py)")

    # Checkpoint
    parser.add_argument("--checkpoint_dir", default="checkpoints/",
                        help="Directory for checkpoints and logs")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")

    # Training hyperparams
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (reduce if OOM)")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--device", default=None,
                        help="Device: cuda / cpu")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")

    args = parser.parse_args()
    train(args)
