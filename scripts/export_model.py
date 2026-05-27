"""Export a training checkpoint to the inference model."""
import argparse
import torch
from pathlib import Path
from hifi_gan import create_generator


def main():
    parser = argparse.ArgumentParser(description="Export checkpoint to tuner.pth")
    parser.add_argument("--checkpoint", default="checkpoints/epoch_005.pt")
    parser.add_argument("--output", default="models/tuner.pth")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    gen = create_generator()
    gen.load_state_dict(ckpt["generator"])
    gen.remove_weight_norm()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(gen.state_dict(), args.output)
    epoch = ckpt.get("epoch", "?")
    print(f"Exported epoch {epoch} checkpoint → {args.output}")


if __name__ == "__main__":
    main()
