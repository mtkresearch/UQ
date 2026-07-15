# Semantic Entropy Probes — Qwen3-8B Run Summary

**Date:** 2026-07-11 / 2026-07-12
**Model:** Qwen3-8B (bf16, flash-attention-2)
**Datasets:** trivia_qa, squad, nq (short-form; bioasq excluded — requires manually-downloaded data file)
**Hardware:** 8× RTX A6000 (48 GB)

## Pipeline

1. **Generation** (`sep-generate`, 8-GPU data-parallel via `slurm/run_multigpu.sh`)
   - 2000 samples per dataset, 1 low-temp (0.1) most-likely answer + 10 high-temp (1.0) samples each.
   - Hidden states saved for the most-likely answer: TBG (`emb_last_tok_before_gen`) and SLT
     (`emb_tok_before_eos`), each `(37, 1, 4096)` — all layers × hidden dim.
   - High-temp samples batched in one `num_return_sequences=10` call, no hidden states (unused downstream).
   - ~10 min generation per dataset across 8 GPUs.
2. **Merge + compute** (`sep-merge-shards`)
   - 8 shards concatenated to exactly 2000 examples per dataset.
   - Semantic entropy via DeBERTa entailment clustering (~50–70 min per dataset, single GPU — the bottleneck).
3. **Probe training** (`sep-train-probes`)
   - Linear logistic-regression probes on SLT hidden states, predicting binarized semantic entropy (SEP)
     and accuracy (Acc probe). ID + cross-dataset OOD evaluation.

## Generation-stage results

| Dataset   | Model accuracy | Semantic Entropy AUROC | p_true (p_false) AUROC |
|-----------|----------------|------------------------|------------------------|
| trivia_qa | 49.4%          | 0.821                  | 0.831                  |
| squad     | 26.4%          | 0.745                  | —                      |
| nq        | 33.2%          | 0.759                  | —                      |

## SEP training results

Best universal SE binarization split: **0.9536**.
Dummy-classifier baselines: trivia_qa 0.72, squad 0.52, nq 0.60.
Auto-selected layer ranges: **SEP [27,32)**, **Acc probe [19,24)**.

AUROC for detecting incorrect answers (higher = better):

| Dataset   | ID SEP | ID Acc probe | OOD SEP | OOD Acc probe |
|-----------|--------|--------------|---------|---------------|
| trivia_qa | 0.758  | 0.747        | 0.727   | 0.643         |
| squad     | 0.735  | 0.687        | 0.656   | 0.645         |
| nq        | 0.739  | 0.718        | 0.691   | 0.655         |

**SEP > Acc-probe OOD win rate: 83.3%** (5 of 6 cross-dataset transfers).

## Key finding

The paper's central claim reproduces on Qwen3-8B: probing hidden states for **semantic entropy**
generalizes across datasets substantially better than probing directly for correctness. The gap is
largest out-of-distribution (e.g. trivia_qa: 0.727 vs 0.643), where the accuracy probe overfits to
dataset-specific correctness cues while the SE probe captures a more transferable uncertainty signal.

## Layerwise figure notes (`figures/se_layerwise_both_tok.pdf`)

- All three datasets: AUROC rises from ~0.5 at layer 0 (embedding, no signal) to a broad high plateau
  from ~layer 15 through ~35 — semantic entropy is linearly decodable in mid-to-late layers.
- Peaks: trivia_qa/nq ~layers 18–22 (~0.85–0.87); squad peaks slightly later and highest (~0.86).
  The auto-selected [27,32) window sits on the plateau for all three (transferable, if marginally
  conservative for trivia_qa/nq).
- TBG and SLT track closely from ~layer 5 onward; SLT is competitive throughout (used for the probes).
- Smooth, stable curves with no collapse or single-layer spikes — confirms clean hidden-state extraction
  across all 37 layers.

## Artifacts

- `models/Qwen3-8B_inference.pkl` — trained SEP + Acc probes with layer ranges.
- `figures/se_layerwise_both_tok.pdf` — per-layer AUROC (TBG vs SLT).
- `figures/{id,ood}_performance_barplot.pdf` — SEP vs Acc-probe comparison.
- Merged generation runs (incl. hidden states) under
  `/build_bak/mtk53686/sep_scratch/shards/Qwen3-8B_{trivia_qa,squad,nq}_*/merged/`.
