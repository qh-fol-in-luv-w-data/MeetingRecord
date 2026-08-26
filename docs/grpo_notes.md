# Stage 3 (GRPO) — not implemented, notes for later

The paper's third finetuning stage uses **Group Relative Policy Optimization**
(GRPO) with rule-based rewards that penalize hallucination and repetition in
the ASR output. This is not implemented in this repo because:

- Fast GRPO training (via `trl.GRPOTrainer`) generates many rollouts per step
  and typically relies on **vLLM** for that generation to be fast enough to
  be practical. vLLM requires CUDA (or ROCm) — it does not run on Apple's MPS
  backend.
- Without vLLM, rollouts would need to go through plain
  `transformers.generate()`, which is slow enough on MPS that a full GRPO run
  would likely take days rather than hours for a small dataset.

## If you want to add this later (on a CUDA machine)

1. Use `trl`'s `GRPOTrainer` (`pip install trl`) starting from the LoRA
   checkpoint produced by `scripts/train_lora.py`.
2. Define a reward function that:
   - Penalizes output length far exceeding reference length (proxy for
     repetition/hallucination).
   - Penalizes repeated n-grams in the hypothesis.
   - Rewards lower WER/CER against the reference transcript (can use `jiwer`).
3. Sample completions with `vllm` for rollout generation, keep the reward
   computation on CPU/GPU as needed.
4. The paper reports this stage moving average tcpMER from ~30.5 (baseline)
   to ~23.7 on their dev set (~22.5 for Vietnamese specifically) — treat this
   as a rough target, not a guarantee, since your data/scale will differ.

Reference: [arXiv:2607.08208](https://arxiv.org/abs/2607.08208)
