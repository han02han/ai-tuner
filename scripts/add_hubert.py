"""
Add HuBERT features to existing training pairs (in-place).

Used after generate_training_data.py was run without --use_hubert.
Reads each pair's shifted.wav, extracts HuBERT features, saves as hubert.npy,
and updates metadata.json with "hubert" field.

Multi-threaded with ThreadPoolExecutor.
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

_HUBERT = None
_DEVICE = "cpu"


def _init_worker():
    global _HUBERT, _DEVICE
    from transformers import HubertModel, Wav2Vec2FeatureExtractor
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _HUBERT = {
        "extractor": Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960"),
        "model": HubertModel.from_pretrained("facebook/hubert-base-ls960").to(_DEVICE).eval(),
    }


def extract_one(pair_dir, p, idx, total):
    global _HUBERT, _DEVICE
    shifted_path = pair_dir / p["shifted"]
    hubert_path = pair_dir / f"{p['id']:05d}_hubert.npy"
    if hubert_path.exists():
        p["hubert"] = hubert_path.name
        return idx, True, ""
    try:
        y, sr = sf.read(str(shifted_path))
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != 16000:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        inputs = _HUBERT["extractor"](y, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            outputs = _HUBERT["model"](**{k: v.to(_DEVICE) for k, v in inputs.items()})
        feats = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        np.save(str(hubert_path), feats.astype(np.float32))
        p["hubert"] = hubert_path.name
        return idx, True, ""
    except Exception as e:
        return idx, False, str(e)


def main():
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/training")
    num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    pair_dir = data_dir / "pairs"
    meta_path = data_dir / "metadata.json"

    with open(meta_path) as f:
        metadata = json.load(f)

    pairs = metadata["pairs"]
    todo = [(i, p) for i, p in enumerate(pairs) if not p.get("hubert")]
    if not todo:
        print("All pairs already have HuBERT features. Nothing to do.")
        return

    print(f"{len(todo)} / {len(pairs)} pairs need HuBERT extraction")
    print(f"Using {num_workers} workers")

    _init_worker()

    with ThreadPoolExecutor(max_workers=num_workers, initializer=_init_worker) as ex:
        futures = {ex.submit(extract_one, pair_dir, p, idx, len(pairs)): idx
                   for idx, p in todo}
        done = 0
        failed = 0
        with tqdm(total=len(todo), desc="Extracting HuBERT") as pbar:
            for future in as_completed(futures):
                idx, ok, err = future.result()
                if ok:
                    done += 1
                else:
                    failed += 1
                    if failed <= 5:
                        print(f"  Failed pair {pairs[idx]['id']}: {err}")
                pbar.update(1)
                if done % 5000 == 0:
                    metadata["pairs"] = pairs
                    with open(meta_path, "w") as f:
                        json.dump(metadata, f, indent=2, ensure_ascii=False)

    metadata["pairs"] = pairs
    metadata["use_hubert"] = True
    metadata["hubert_model"] = "facebook/hubert-base-ls960"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    usable = sum(1 for p in pairs if p.get("hubert"))
    print(f"Done: {usable}/{len(pairs)} pairs have HuBERT ({failed} failed)")


if __name__ == "__main__":
    main()
