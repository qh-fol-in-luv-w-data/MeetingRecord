#!/usr/bin/env python3
"""End-to-end inference: diarize an audio file, then transcribe each speaker
turn with Qwen3-ASR (optionally with a LoRA adapter applied).

Usage:
    python scripts/infer.py --audio path/to/audio.wav
    python scripts/infer.py --audio path/to/audio.wav --lora-checkpoint checkpoints/qwen3-asr-vi-lora
"""
import argparse

import soundfile as sf
import torch
from pyannote.audio import Pipeline
from transformers import AutoModel, AutoProcessor


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def format_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--lora-checkpoint", default=None, help="Optional LoRA adapter dir")
    parser.add_argument("--num-speakers", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    print("Loading diarization pipeline...")
    diar_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    diar_pipeline.to(torch.device(device))

    print(f"Loading ASR model: {args.base_model}")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.base_model, trust_remote_code=True)

    if args.lora_checkpoint:
        from peft import PeftModel

        print(f"Applying LoRA adapter from {args.lora_checkpoint}")
        model = PeftModel.from_pretrained(model, args.lora_checkpoint)

    model.to(device)
    model.eval()

    print("Running diarization...")
    diarization = diar_pipeline(args.audio, num_speakers=args.num_speakers)

    print("Loading audio for per-turn transcription...")
    audio, sr = sf.read(args.audio)

    print("\nTranscript:\n")
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_sample = int(turn.start * sr)
        end_sample = int(turn.end * sr)
        segment = audio[start_sample:end_sample]
        if len(segment) == 0:
            continue

        inputs = processor(audio=[segment], sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"[{speaker}] {format_ts(turn.start)} - {format_ts(turn.end)}  {text}")


if __name__ == "__main__":
    main()
