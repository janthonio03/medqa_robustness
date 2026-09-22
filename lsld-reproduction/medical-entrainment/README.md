# Medical hard-distractor entrainment experiments

This directory preserves the experiment-code snapshot used for the medical hard-distractor analyses performed after the base LSLD reproduction.

It is intentionally kept separate from the earlier `lsld-reproduction` v0.4 tree because these medical experiments were run from a later v1.0.0 server codebase. The files here are the result-producing scripts and modified runtime files supplied from that experiment environment; they are not intended to replace the older reproduction tree in place.

## Included code

- `scripts/run_medqa_head_search.py` — adapts the LSLD head-mask search to the 89/11/11 medical split and supports microbatch gradient accumulation.
- `scripts/evaluate_medqa_all111_mask.py` — applies the selected joint head mask to all 111 clean-to-hard target-flip examples and evaluates restricted A/B/C/D logits.
- `scripts/scan_medqa_entrainment_heads.py` — performs single-head ablation and computes per-head/per-example entrainment effects.
- `scripts/scan_medqa_all111_heads.py` — exhaustively evaluates all 32 x 32 = 1024 attention heads on all 111 examples.
- `src/lsld_repro/paper_head.py` — experiment-driver snapshot used by the medical runs, including opt-in gradient accumulation.
- `src/lsld_repro/official_code/head_search/circuit_lms/hooked_transformers.py` — runtime snapshot with the opt-in last-position-only unembedding path used for long medical prompts.

## Core protocol represented by these files

- Base model: `meta-llama/Llama-3.1-8B`
- Medical analysis set: 111 examples for which the clean prompt selects the gold answer and the hard-distractor prompt selects the intended hard target.
- Initial split for joint head search: 89 train / 11 dev / 11 test.
- Multiple-choice evaluation: next-token A/B/C/D logit comparison at the final prompt position after `Answer:`.
- Single-head analysis: ablate one attention head at a time while keeping the other 1023 heads active.
- Entrainment score: reduction in the clean-to-hard Gold-vs-Target preference shift after head ablation.

## Excluded intentionally

Datasets, checkpoints, learned masks, generated outputs, cache files, OOM/debug artifacts, backup files, and ad-hoc result-filtering snippets are not committed here.

This directory is a compact provenance snapshot for the medical experiment results rather than a standalone environment export.
