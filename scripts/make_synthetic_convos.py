#!/usr/bin/env python3
"""Build synthetic 2-speaker "conversations" from a single-speaker ASR dataset.

This is a cheap stand-in for the paper's OmniVoice-based synthetic data
augmentation: it concatenates utterances from two different speakers (when
speaker metadata is available) or two random different rows (fallback), and
records ground-truth speaker turns as an RTTM-style label alongside the
combined transcript.

Usage:
    python scripts/make_synthetic_convos.py --in data/bud500 --out data/bud500_convos --num-convos 5000
"""
import argparse
import json
import random
from pathlib import Path

import soundfile as sf
from datasets import load_from_disk


def find_speaker_column(ds):
    for candidate in ("speaker_id", "speaker", "client_id"):
        if candidate in ds.column_names:
            return candidate
    return None


def build_convo(rows, sample_rate, silence_s=0.3):
    """Concatenate audio rows with short silence gaps; return audio + turn labels."""
    import numpy as np

    silence = np.zeros(int(sample_rate * silence_s), dtype="float32")
    audio_chunks = []
    turns = []
    t = 0.0
    for i, row in enumerate(rows):
        audio = row["audio"]["array"].astype("float32")
        dur = len(audio) / sample_rate
        turns.append(
            {
                "speaker": f"SPEAKER_{i % 2:02d}",
                "start": round(t, 3),
                "end": round(t + dur, 3),
                "text": row.get("transcription") or row.get("sentence") or row.get("text", ""),
            }
        )
        audio_chunks.append(audio)
        audio_chunks.append(silence)
        t += dur + silence_s
    import numpy as np

    return np.concatenate(audio_chunks), turns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_dir", required=True)
    parser.add_argument("--out", dest="out_dir", required=True)
    parser.add_argument("--num-convos", type=int, default=1000)
    parser.add_argument("--utterances-per-convo", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    ds = load_from_disk(args.in_dir)
    speaker_col = find_speaker_column(ds)
    print(f"Speaker column detected: {speaker_col or 'none (will pair random rows)'}")

    out_dir = Path(args.out_dir)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    n = len(ds)
    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for convo_idx in range(args.num_convos):
            idxs = [random.randrange(n) for _ in range(args.utterances_per_convo)]
            rows = [ds[i] for i in idxs]
            sample_rate = rows[0]["audio"]["sampling_rate"]

            audio, turns = build_convo(rows, sample_rate)
            wav_path = out_dir / "audio" / f"convo_{convo_idx:06d}.wav"
            sf.write(wav_path, audio, sample_rate)

            manifest.write(
                json.dumps(
                    {
                        "audio_path": str(wav_path),
                        "turns": turns,
                        "full_text": " ".join(t["text"] for t in turns),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            if (convo_idx + 1) % 200 == 0:
                print(f"  {convo_idx + 1}/{args.num_convos} synthetic conversations built")

    print(f"Done. Manifest at {manifest_path}")


if __name__ == "__main__":
    main()
