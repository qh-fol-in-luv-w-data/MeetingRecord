#!/usr/bin/env python3
"""LoRA finetuning of Qwen3-ASR on a Vietnamese ASR dataset (Stage-2-style,
per the paper's approach, minus the OmniVoice-specific augmentation).

Works on MPS (Apple Silicon), CUDA, or CPU. Start with the 0.6B model on MPS.

Expects a manifest.jsonl produced by scripts/make_synthetic_convos.py, or any
jsonl with {"audio_path": ..., "full_text": ...} rows -- for plain
single-utterance finetuning (no synthetic conversations), point --data at a
jsonl with the same two fields per row.

Usage:
    python scripts/train_lora.py \
        --base-model Qwen/Qwen3-ASR-0.6B \
        --data data/bud500_convos/manifest.jsonl \
        --output-dir checkpoints/qwen3-asr-vi-lora \
        --device auto --epochs 3 --batch-size 4 --lr 1e-4

NOTE: Qwen3-ASR's exact processor/model class names may shift between
`transformers` releases. If `AutoModel`/`AutoProcessor` with
`trust_remote_code=True` doesn't resolve the right classes for the version
you have installed, check the model card at
https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf for the exact classes and swap
them in here. If LoRA target module names below don't match, run
`print(model)` once and adjust `--lora-target-modules`.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModel,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class JsonlAudioTextDataset(Dataset):
    def __init__(self, manifest_path, processor, max_target_length=256):
        self.rows = []
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        return {
            "audio_path": row["audio_path"],
            "text": row.get("full_text") or row.get("text", ""),
        }


def make_collate_fn(processor, max_target_length):
    import soundfile as sf

    def collate(batch):
        audios = []
        sample_rates = []
        for item in batch:
            audio, sr = sf.read(item["audio_path"])
            audios.append(audio)
            sample_rates.append(sr)

        texts = [item["text"] for item in batch]

        inputs = processor(
            audio=audios,
            sampling_rate=sample_rates[0],
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_target_length,
        )
        # Most seq2seq-style processors return `labels` directly when given
        # `text=`. If yours doesn't, derive labels from input_ids/text tokens
        # here instead.
        if "labels" not in inputs:
            inputs["labels"] = inputs["input_ids"].clone()
        return inputs

    return collate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--data", required=True, help="Path to manifest.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated module names to apply LoRA to. Adjust after "
        "inspecting `print(model)` if these don't match Qwen3-ASR's layer names.",
    )
    parser.add_argument("--max-target-length", type=int, default=256)
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    print(f"Loading base model + processor: {args.base_model}")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.base_model, trust_remote_code=True)
    model.to(device)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules.split(","),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = JsonlAudioTextDataset(args.data, processor, args.max_target_length)
    print(f"Loaded {len(dataset)} training examples from {args.data}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        report_to=[],
        # `use_mps_device` is implicit via model.to("mps") + no CUDA-only flags below.
        fp16=False,  # MPS does not support fp16 training reliably; leave off.
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=make_collate_fn(processor, args.max_target_length),
    )

    trainer.train()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
