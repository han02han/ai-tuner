"""
Generate paired training data for neural pitch correction using WORLD vocoder
+ HuBERT content features.

Takes clean vocal audio files and produces paired (out_of_tune, clean)
training examples by applying controlled random pitch deviations.

WORLD analysis (expensive) is done once per input file. HuBERT features are
extracted once per file and sliced per segment.

Supports multi-threaded file processing via --num_workers. pyworld C functions
release the GIL, so I/O and WORLD synthesis scale across threads.
"""

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyworld as pw
import soundfile as sf
import threading
import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor


def _apply_pitch_shift(
    f0_seg: np.ndarray, 
    sp_seg: np.ndarray,
    shift_cents: float,
    formant_shift_ratio: float,
) -> np.ndarray:
    """Apply pitch shift to copied WORLD params. Returns shifted f0."""
    f0_new = f0_seg.copy()
    if abs(shift_cents) < 0.1:
        return f0_new, sp_seg
    factor = 2.0 ** (shift_cents / 1200.0)
    voiced = f0_seg > 0
    f0_new[voiced] *= factor
    if formant_shift_ratio > 0:
        env_ratio = 1.0 + (factor - 1.0) * formant_shift_ratio
        n_bins = sp_seg.shape[1]
        src = np.arange(n_bins) / env_ratio
        lo = np.clip(np.floor(src).astype(np.intp), 0, n_bins - 1)
        hi = np.clip(lo + 1, 0, n_bins - 1)
        frac = src - lo.astype(np.float64)
        sp_new = sp_seg.copy()
        for i in np.where(voiced)[0]:
            sp_new[i] = sp_seg[i][lo] * (1 - frac) + sp_seg[i][hi] * frac
        return f0_new, sp_new
    return f0_new, sp_seg


# ---------------------------------------------------------------------------
# HuBERT feature extraction (global singleton to avoid reload)
# ---------------------------------------------------------------------------

_HUBERT_CACHE: dict = {}
_HUBERT_LOCK = threading.Lock()


def _get_hubert_model(device="cpu"):
    """Load HuBERT model lazily (cached after first call, thread-safe)."""
    key = f"hubert::{device}"
    if key not in _HUBERT_CACHE:
        with _HUBERT_LOCK:
            if key not in _HUBERT_CACHE:  # double-check
                _HUBERT_CACHE["extractor"] = Wav2Vec2FeatureExtractor.from_pretrained(
                    "facebook/hubert-base-ls960")
                _HUBERT_CACHE[key] = HubertModel.from_pretrained(
                    "facebook/hubert-base-ls960").to(device).eval()
    return _HUBERT_CACHE["extractor"], _HUBERT_CACHE[key]


def _extract_hubert_features(y: np.ndarray, sr: int, device="cpu") -> np.ndarray:
    """Extract HuBERT features from audio.

    Args:
        y: audio waveform (float32, mono)
        sr: sample rate (must be 16000 for base HuBERT)

    Returns:
        features: (T, 768) HuBERT features at 50 Hz frame rate
    """
    extractor, model = _get_hubert_model(device)
    # HuBERT expects 16kHz
    if sr != 16000:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    inputs = extractor(y, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**{k: v.to(device) for k, v in inputs.items()})
    return outputs.last_hidden_state.squeeze(0).cpu().numpy()


def _process_file(args):
    """Process a single audio file: WORLD analysis + generate pairs.

    Runs in a worker thread. Returns (file_name, pairs_metadata, pair_count).
    """
    (audio_file, pairs_dir, base_idx, pairs_per_file, max_shift_cents,
     target_sr, max_duration_sec, formant_shift_ratio, min_clean_ratio,
     use_hubert) = args
    pairs_dir = Path(pairs_dir)

    try:
        y, sr = sf.read(str(audio_file))
    except Exception as e:
        print(f"  Skipping {audio_file.name}: {e}")
        return audio_file.name, [], 0

    if y.ndim > 1:
        y = y.mean(axis=1)

    if sr != target_sr:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    if len(y) < sr:
        return audio_file.name, [], 0

    # WORLD analysis ONCE per file
    y_f64 = y.astype(np.float64)
    try:
        f0, t = pw.dio(y_f64, sr, f0_floor=65.0, f0_ceil=1047.0, frame_period=5.0)
        f0 = pw.stonemask(y_f64, f0, t, sr)
        sp = pw.cheaptrick(y_f64, f0, t, sr)
        ap = pw.d4c(y_f64, f0, t, sr)
    except Exception:
        return audio_file.name, [], 0

    n_frames = len(f0)
    total_samples = len(y)
    frame_period_samples = int(target_sr * 0.005)

    pairs_metadata = []
    pair_idx = 0
    failed = 0

    for _ in range(pairs_per_file):
        segment_dur = random.uniform(3.0, min(max_duration_sec, total_samples / sr))
        seg_samples = int(segment_dur * sr)

        if seg_samples >= total_samples:
            start_sample = 0
        else:
            start_sample = random.randint(0, total_samples - seg_samples)

        end_sample = start_sample + seg_samples
        segment_clean = y[start_sample:end_sample]

        start_frame = start_sample // frame_period_samples
        end_frame = min(n_frames, (end_sample + frame_period_samples - 1) // frame_period_samples)
        if end_frame <= start_frame:
            failed += 1
            continue

        if random.random() < min_clean_ratio:
            shift_cents = 0.0
        else:
            sign = random.choice([-1, 1])
            shift_cents = sign * random.uniform(20.0, max_shift_cents)

        try:
            f0_seg = f0[start_frame:end_frame]
            sp_seg = sp[start_frame:end_frame]
            ap_seg = ap[start_frame:end_frame]

            f0_shifted, sp_shifted = _apply_pitch_shift(
                f0_seg, sp_seg, shift_cents, formant_shift_ratio)

            y_shifted = pw.synthesize(f0_shifted, sp_shifted, ap_seg, sr)
            y_shifted = y_shifted.astype(np.float32)

            min_len = min(len(y_shifted), len(segment_clean))
            y_shifted = y_shifted[:min_len]
            y_clean = segment_clean[:min_len].astype(np.float32)

            global_id = base_idx + pair_idx
            clean_path = pairs_dir / f"{global_id:05d}_clean.wav"
            shifted_path = pairs_dir / f"{global_id:05d}_shifted.wav"
            f0_path = pairs_dir / f"{global_id:05d}_f0.npy"

            sf.write(str(clean_path), y_clean, sr)
            sf.write(str(shifted_path), y_shifted, sr)
            np.save(str(f0_path), f0_seg.astype(np.float32))

            hubert_path_str = ""
            if use_hubert:
                hubert_path = pairs_dir / f"{global_id:05d}_hubert.npy"
                try:
                    hubert_feats = _extract_hubert_features(y_shifted, sr, device="cuda" if torch.cuda.is_available() else "cpu")
                    np.save(str(hubert_path), hubert_feats.astype(np.float32))
                    hubert_path_str = str(hubert_path.name)
                except Exception as e:
                    print(f"  HuBERT extraction failed for pair {global_id}: {e}")

            pairs_metadata.append({
                "id": global_id,
                "clean": str(clean_path.name),
                "shifted": str(shifted_path.name),
                "f0": str(f0_path.name),
                "hubert": hubert_path_str if use_hubert else "",
                "source_file": audio_file.name,
                "duration_sec": round(min_len / sr, 2),
            })
            pair_idx += 1

        except Exception:
            failed += 1
            continue

    if failed:
        print(f"  {audio_file.name}: {failed} pairs failed")

    return audio_file.name, pairs_metadata, pair_idx


def generate_dataset(
    input_dir: str,
    output_dir: str,
    pairs_per_file: int = 20,
    max_shift_cents: float = 300.0,
    target_sr: int = 22050,
    max_duration_sec: float = 15.0,
    formant_shift_ratio: float = 0.4,
    min_clean_ratio: float = 0.05,
    num_workers: int = 1,
    use_hubert: bool = True,
):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    pairs_dir = output_path / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    audio_extensions = ["*.wav", "*.mp3", "*.flac"]
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(sorted(input_path.rglob(ext)))

    if not audio_files:
        print(f"No audio files found in {input_dir}")
        print("Place clean vocal WAV/MP3/FLAC files in data/clean/ and re-run.")
        return

    print(f"Found {len(audio_files)} audio files")
    print(f"Using {num_workers} workers (multi-threaded, pyworld releases GIL)")

    # Assign base index to each file so workers write non-conflicting filenames
    file_args = []
    for i, audio_file in enumerate(audio_files):
        file_args.append((
            audio_file, str(pairs_dir), i * pairs_per_file,
            pairs_per_file, max_shift_cents, target_sr,
            max_duration_sec, formant_shift_ratio, min_clean_ratio,
            use_hubert,
        ))

    metadata = {
        "total_pairs": 0,
        "sample_rate": target_sr,
        "max_shift_cents": max_shift_cents,
        "vocoder": "WORLD",
        "formant_shift_ratio": formant_shift_ratio,
        "use_hubert": use_hubert,
        "hubert_model": "facebook/hubert-base-ls960" if use_hubert else "",
        "source_files": [str(f.name) for f in audio_files],
        "pairs": [],
    }

    total_pairs = 0
    completed = 0
    failed_files = 0

    if num_workers <= 1:
        # Single-threaded path (avoids thread overhead for --num_workers 1)
        from tqdm import tqdm as _tqdm
        for args_tuple in _tqdm(file_args, desc="Processing files"):
            fname, pairs, count = _process_file(args_tuple)
            metadata["pairs"].extend(pairs)
            total_pairs += count
            completed += 1
            if count == 0:
                failed_files += 1
    else:
        from tqdm import tqdm as _tqdm
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_file, a): a[0].name for a in file_args}
            with _tqdm(total=len(file_args), desc="Processing files") as pbar:
                for future in as_completed(futures):
                    fname, pairs, count = future.result()
                    metadata["pairs"].extend(pairs)
                    total_pairs += count
                    completed += 1
                    if count == 0:
                        failed_files += 1
                    pbar.set_postfix(pairs=total_pairs, failed=failed_files)
                    pbar.update(1)

    # Sort pairs by id for consistent ordering
    metadata["pairs"].sort(key=lambda p: p["id"])
    metadata["total_pairs"] = total_pairs

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated {total_pairs} training pairs in {output_path}")
    print(f"  Processed {completed}/{len(audio_files)} files"
          f"{f' ({failed_files} failed)' if failed_files else ''}")
    print(f"  Clean audio  → {pairs_dir}/XXXXX_clean.wav")
    print(f"  Shifted audio → {pairs_dir}/XXXXX_shifted.wav")
    print(f"  Metadata     → {output_path}/metadata.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate training data for neural tuner (WORLD-based)"
    )
    parser.add_argument("--input_dir", default="data/clean/",
                        help="Directory of clean vocal audio files")
    parser.add_argument("--output_dir", default="data/training/",
                        help="Output directory for paired data")
    parser.add_argument("--pairs_per_file", type=int, default=20,
                        help="Number of variations per input file")
    parser.add_argument("--max_shift_cents", type=float, default=300.0,
                        help="Maximum pitch deviation in cents")
    parser.add_argument("--target_sr", type=int, default=22050,
                        help="Target sample rate")
    parser.add_argument("--max_duration_sec", type=float, default=15.0,
                        help="Maximum segment duration in seconds")
    parser.add_argument("--formant_shift_ratio", type=float, default=0.4,
                        help="Formant coupling ratio (0.0=pure f0 shift, "
                             "0.4=moderate, 1.0=full coupling)")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel worker threads")
    parser.add_argument("--use_hubert", action="store_true", default=True,
                        help="Extract HuBERT features for each pair (requires transformers)")
    args = parser.parse_args()

    generate_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pairs_per_file=args.pairs_per_file,
        max_shift_cents=args.max_shift_cents,
        target_sr=args.target_sr,
        max_duration_sec=args.max_duration_sec,
        formant_shift_ratio=args.formant_shift_ratio,
        num_workers=args.num_workers,
        use_hubert=args.use_hubert,
    )
