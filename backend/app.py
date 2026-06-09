"""FastAPI backend for AI Tuner - pitch correction web service."""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import sys
import uuid
from pathlib import Path

# Ensure backend/ is on the Python path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from typing import Optional
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from tuner import tune_to_scale, tune_to_reference, tune_neural, tune_compare

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="AI Tuner", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SCALE_OPTIONS = [
    "major", "minor", "harmonic_minor", "melodic_minor",
    "pentatonic_major", "pentatonic_minor", "chromatic",
]
KEY_OPTIONS = ["auto", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
RHYTHM_OPTIONS = ["none", "grid", "reference"]


@app.get("/api/scales")
def list_scales():
    return SCALE_OPTIONS


@app.get("/api/keys")
def list_keys():
    return KEY_OPTIONS


@app.get("/api/rhythms")
def list_rhythms():
    return RHYTHM_OPTIONS


@app.post("/api/tune/scale")
def tune_by_scale(
    audio: UploadFile = File(...),
    key: str = Form("auto"),
    scale: str = Form("major"),
    strength: float = Form(1.0),
    rhythm: str = Form("none"),
    rhythm_ref: Optional[UploadFile] = File(None),
    rhythm_bpm: Optional[float] = Form(None),
):
    """Pitch correction using musical scale."""
    stem = Path(audio.filename).stem if audio.filename else "audio"
    input_path = UPLOAD_DIR / f"{uuid.uuid4()}.wav"
    out_name = f"{stem}_tuned_dsp_scale.wav"
    output_path = OUTPUT_DIR / out_name
    rhythm_ref_path = None

    try:
        input_path.write_bytes(audio.file.read())

        if rhythm == "reference" and rhythm_ref is not None:
            rhythm_ref_path = UPLOAD_DIR / f"{uuid.uuid4()}_rhythm_ref.wav"
            rhythm_ref_path.write_bytes(rhythm_ref.file.read())
            rhythm_ref_path = str(rhythm_ref_path)

        result = tune_to_scale(
            str(input_path),
            key=key,
            scale_type=scale,
            correction_strength=strength,
            rhythm_mode=rhythm,
            rhythm_ref_path=rhythm_ref_path,
            rhythm_bpm=rhythm_bpm,
        )

        import soundfile as sf
        sf.write(str(output_path), result.audio, result.sample_rate)

        return {
            "method": result.method,
            "key_detected": result.key_detected,
            "frames_processed": result.frames_processed,
            "frames_corrected": result.frames_corrected,
            "avg_correction_cents": round(result.avg_correction_cents, 1),
            "download_url": f"/api/download/{output_path.name}",
        }
    finally:
        if input_path.exists():
            input_path.unlink()
        if rhythm_ref_path and Path(rhythm_ref_path).exists():
            Path(rhythm_ref_path).unlink()


@app.post("/api/tune/neural")
def tune_by_neural(
    audio: UploadFile = File(...),
    key: str = Form("auto"),
    scale: str = Form("major"),
    strength: float = Form(1.0),
    rhythm: str = Form("none"),
    rhythm_ref: Optional[UploadFile] = File(None),
    rhythm_bpm: Optional[float] = Form(None),
):
    """Pitch correction using neural vocoder (requires trained model)."""
    stem = Path(audio.filename).stem if audio.filename else "audio"
    input_path = UPLOAD_DIR / f"{uuid.uuid4()}.wav"
    out_name = f"{stem}_tuned_neural.wav"
    output_path = OUTPUT_DIR / out_name
    rhythm_ref_path = None

    try:
        input_path.write_bytes(audio.file.read())

        if rhythm == "reference" and rhythm_ref is not None:
            rhythm_ref_path = UPLOAD_DIR / f"{uuid.uuid4()}_rhythm_ref.wav"
            rhythm_ref_path.write_bytes(rhythm_ref.file.read())
            rhythm_ref_path = str(rhythm_ref_path)

        result = tune_neural(
            str(input_path),
            key=key,
            scale_type=scale,
            correction_strength=strength,
            rhythm_mode=rhythm,
            rhythm_ref_path=rhythm_ref_path,
            rhythm_bpm=rhythm_bpm,
        )

        import soundfile as sf
        sf.write(str(output_path), result.audio, result.sample_rate)

        return {
            "method": result.method,
            "key_detected": result.key_detected,
            "frames_processed": result.frames_processed,
            "frames_corrected": result.frames_corrected,
            "avg_correction_cents": round(result.avg_correction_cents, 1),
            "download_url": f"/api/download/{output_path.name}",
        }
    except FileNotFoundError as e:
        return {"error": str(e), "neural_available": False}, 503
    finally:
        if input_path.exists():
            input_path.unlink()
        if rhythm_ref_path and Path(rhythm_ref_path).exists():
            Path(rhythm_ref_path).unlink()


@app.post("/api/tune/compare")
def tune_comparison(
    audio: UploadFile = File(...),
    key: str = Form("auto"),
    scale: str = Form("major"),
    strength: float = Form(1.0),
    rhythm: str = Form("none"),
    rhythm_ref: Optional[UploadFile] = File(None),
    rhythm_bpm: Optional[float] = Form(None),
):
    """Run both DSP and neural correction, return both results for comparison."""
    stem = Path(audio.filename).stem if audio.filename else "audio"
    input_path = UPLOAD_DIR / f"{uuid.uuid4()}.wav"
    rhythm_ref_path = None

    try:
        input_path.write_bytes(audio.file.read())

        if rhythm == "reference" and rhythm_ref is not None:
            rhythm_ref_path = UPLOAD_DIR / f"{uuid.uuid4()}_rhythm_ref.wav"
            rhythm_ref_path.write_bytes(rhythm_ref.file.read())
            rhythm_ref_path = str(rhythm_ref_path)

        results = tune_compare(
            str(input_path),
            key=key,
            scale_type=scale,
            correction_strength=strength,
            rhythm_mode=rhythm,
            rhythm_ref_path=rhythm_ref_path,
            rhythm_bpm=rhythm_bpm,
        )

        out = {"neural_available": results["neural_available"]}

        # Save DSP result
        dsp_path = OUTPUT_DIR / f"{stem}_compare_dsp.wav"
        import soundfile as sf
        sf.write(str(dsp_path), results["dsp"].audio, results["dsp"].sample_rate)
        out["dsp"] = {
            "method": "dsp",
            "key_detected": results["dsp"].key_detected,
            "frames_processed": results["dsp"].frames_processed,
            "frames_corrected": results["dsp"].frames_corrected,
            "avg_correction_cents": round(results["dsp"].avg_correction_cents, 1),
            "download_url": f"/api/download/{dsp_path.name}",
        }

        # Save neural result (if available)
        if results["neural"] is not None:
            nn_path = OUTPUT_DIR / f"{stem}_compare_neural.wav"
            sf.write(str(nn_path), results["neural"].audio, results["neural"].sample_rate)
            out["neural"] = {
                "method": "neural",
                "key_detected": results["neural"].key_detected,
                "frames_processed": results["neural"].frames_processed,
                "frames_corrected": results["neural"].frames_corrected,
                "avg_correction_cents": round(results["neural"].avg_correction_cents, 1),
                "download_url": f"/api/download/{nn_path.name}",
            }

        return out
    finally:
        if input_path.exists():
            input_path.unlink()


@app.post("/api/tune/reference")
def tune_by_ref(
    audio: UploadFile = File(...),
    reference: UploadFile = File(...),
    strength: float = Form(1.0),
    rhythm: str = Form("none"),
):
    """Pitch correction using a reference (original singer) audio."""
    stem = Path(audio.filename).stem if audio.filename else "audio"
    input_path = UPLOAD_DIR / f"{uuid.uuid4()}.wav"
    ref_path = UPLOAD_DIR / f"{uuid.uuid4()}_ref.wav"
    out_name = f"{stem}_tuned_dsp_ref.wav"
    output_path = OUTPUT_DIR / out_name

    try:
        input_path.write_bytes(audio.file.read())
        ref_path.write_bytes(reference.file.read())

        result = tune_to_reference(
            str(input_path),
            str(ref_path),
            correction_strength=strength,
            rhythm_mode=rhythm,
        )

        import soundfile as sf
        sf.write(str(output_path), result.audio, result.sample_rate)

        return {
            "method": result.method,
            "key_detected": result.key_detected,
            "frames_processed": result.frames_processed,
            "frames_corrected": result.frames_corrected,
            "avg_correction_cents": round(result.avg_correction_cents, 1),
            "download_url": f"/api/download/{output_path.name}",
        }
    finally:
        if input_path.exists():
            input_path.unlink()
        if ref_path.exists():
            ref_path.unlink()


@app.get("/api/download/{filename}")
def download_file(filename: str):
    """Download a processed audio file."""
    path = OUTPUT_DIR / filename
    if not path.exists():
        return {"error": "file not found"}, 404
    return FileResponse(str(path), media_type="audio/wav", filename=f"tuned_{filename}")


# Serve frontend
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8050, reload=True)
