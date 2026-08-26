# MeetingRecorder

Diarization + Vietnamese ASR finetuning pipeline, inspired by
["Diarization-Guided Qwen-ASR Adaptation for Multilingual Two-Speaker Conversational Speech"](https://arxiv.org/abs/2607.08208).

This repo implements a **scaled-down, reproducible** version of the paper's
approach that runs on a single machine (including Apple Silicon via MPS):

1. **Diarization frontend** — VAD + speaker embeddings + clustering (2 speakers),
   using `pyannote.audio` instead of the paper's 3D-Speaker/CAMPPlus stack
   (functionally equivalent, easier to install).
2. **LoRA finetuning of Qwen3-ASR** on Vietnamese speech data.

The paper's **Stage 3 (GRPO reinforcement learning)** is documented but **not
implemented** here — it depends on vLLM for fast rollout generation, which
does not support Apple's MPS backend (CUDA/ROCm only). See
[`docs/grpo_notes.md`](docs/grpo_notes.md) for what it would take to add it on
a CUDA machine.

## What's NOT reproduced from the paper

- The paper trains on ~1,500 hours across 21 languages with data augmented by
  **OmniVoice** (a zero-shot voice cloning model) — that model/data isn't
  public. This repo uses **public Vietnamese datasets only** (see below).
- No GRPO / RL stage (see above).
- Results will not match the paper's tcpMER numbers — this is a starting
  point for your own experiments, not a reproduction.

## Requirements

- Python 3.10+
- macOS with Apple Silicon (M1/M2/M3/M4) for MPS, or a CUDA GPU, or CPU
  (slow) — the training script auto-detects `mps` / `cuda` / `cpu`
- ~30GB free disk for the datasets below
- `ffmpeg` installed (`brew install ffmpeg` on macOS)
- A free [Hugging Face account + token](https://huggingface.co/settings/tokens)
  (`huggingface-cli login`) — needed to download Qwen3-ASR and some datasets
- A free [pyannote.audio](https://huggingface.co/pyannote/speaker-diarization-3.1)
  gated-model acceptance on Hugging Face (click "agree" on the model page)

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training data

This repo uses **public** Vietnamese ASR datasets — pick one or combine them.

| Dataset | Hours | License | Source |
|---|---|---|---|
| **Bud500** | ~500h | Apache 2.0 | [`linhtran92/viet_bud500`](https://huggingface.co/datasets/linhtran92/viet_bud500) |
| **VIVOS** | ~15h | Free for research | [`vivos`](https://huggingface.co/datasets/vivos) |
| **Common Voice (vi)** | varies by release | CC0 | [Mozilla Common Voice](https://commonvoice.mozilla.org/vi/datasets) |
| **doof-ferb Vietnamese speech collection** | aggregated | mixed, check each | [HF collection](https://huggingface.co/collections/doof-ferb/vietnamese-speech-dataset-65c6af8c15c9950537862fa6) |

Download the default (Bud500) with:

```bash
python scripts/download_data.py --dataset bud500 --out data/bud500
```

### Simulating 2-speaker conversations (for diarization-aware training)

The public datasets above are single-speaker utterances. To approximate the
paper's two-speaker conversational setting, `scripts/make_synthetic_convos.py`
concatenates/overlaps pairs of utterances from different speakers into
synthetic "conversations" with known speaker turns — a cheap stand-in for the
paper's OmniVoice-based augmentation.

```bash
python scripts/make_synthetic_convos.py \
  --in data/bud500 \
  --out data/bud500_convos \
  --num-convos 5000
```

## Pipeline

### 1. Diarization frontend (inference only, no training needed)

```bash
python scripts/diarize.py --audio path/to/audio.wav --out path/to/audio.rttm
```

Uses `pyannote/speaker-diarization-3.1`, clustering fixed to 2 speakers to
match the paper's setup. Requires accepting the model's gated terms on
Hugging Face once.

### 2. LoRA finetune Qwen3-ASR

```bash
python scripts/train_lora.py \
  --base-model Qwen/Qwen3-ASR-0.6B \
  --data data/bud500_convos \
  --output-dir checkpoints/qwen3-asr-vi-lora \
  --device auto \
  --epochs 3 \
  --batch-size 4 \
  --lr 1e-4
```

Notes for MPS (Apple Silicon):

- Start with the **0.6B** model, not 1.7B — unified memory pressure adds up
  fast without CUDA-only optimizations (8-bit optimizers, flash-attention,
  DeepSpeed are unavailable on MPS).
- If you hit OOM, lower `--batch-size` and raise
  `--gradient-accumulation-steps` instead.
- Expect training to be several times slower than an equivalent Nvidia GPU.

### 3. Run inference (diarize + transcribe)

```bash
python scripts/infer.py --audio path/to/audio.wav --lora-checkpoint checkpoints/qwen3-asr-vi-lora
```

Outputs a transcript with speaker labels and timestamps, e.g.:

```
[SPEAKER_00] 00:00:00.12 - 00:00:03.40  Xin chào, hôm nay chúng ta bàn về việc gì?
[SPEAKER_01] 00:00:03.60 - 00:00:07.05  Chào bạn, mình muốn hỏi về...
```

## Project layout

```
.
├── README.md
├── requirements.txt
├── docs/
│   └── grpo_notes.md          # what Stage 3 (GRPO) would need, not implemented
├── scripts/
│   ├── download_data.py       # fetch public VN datasets from HF
│   ├── make_synthetic_convos.py  # build 2-speaker synthetic conversations
│   ├── diarize.py             # pyannote-based diarization frontend
│   ├── train_lora.py          # Stage-2-style LoRA finetuning
│   └── infer.py               # end-to-end diarize + transcribe
└── data/                      # datasets land here (gitignored)
```

## Reference

Wu, H., Han, R. et al. *Diarization-Guided Qwen-ASR Adaptation for
Multilingual Two-Speaker Conversational Speech*.
[arXiv:2607.08208](https://arxiv.org/abs/2607.08208)
