#!/usr/bin/env python3
"""Speaker diarization frontend using pyannote.audio, fixed to 2 speakers.

Requires:
  - `pip install pyannote.audio`
  - A Hugging Face token with access accepted for
    pyannote/speaker-diarization-3.1 (visit the model page and click "agree").
  - `huggingface-cli login` run once beforehand.

Usage:
    python scripts/diarize.py --audio path/to/audio.wav --out path/to/audio.rttm
"""
import argparse

import torch
from pyannote.audio import Pipeline


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out", required=True, help="Output RTTM file path")
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=2,
        help="Fixed speaker count, matching the paper's two-speaker setting",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    pipeline.to(device)

    diarization = pipeline(args.audio, num_speakers=args.num_speakers)

    with open(args.out, "w") as f:
        diarization.write_rttm(f)

    print(f"Diarization written to {args.out}")
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"  [{speaker}] {turn.start:.2f}s - {turn.end:.2f}s")


if __name__ == "__main__":
    main()
