# Llama See, Llama Do — paper-first reproduction

This project reproduces the controlled contextual-entrainment and attention-head
experiments in Niu et al. (ACL 2025), *Llama See, Llama Do*. Version 0.4 adds a
strict published-result mode: it preserves the released data RNG stream and
reproduces the final-development-batch behavior that affected checkpoint
selection in the public implementation.

Primary references:

- Paper: <https://aclanthology.org/2025.acl-long.791/>
- Official code: <https://github.com/frankniujc/entrainment>
- Audited official commit: `c22d893dd358b27c057ef6ba37c2da1101866497`

## Paper-reproduction defaults

| Component | v0.4 paper-first behavior |
|---|---|
| Model | `meta-llama/Llama-3.1-8B` |
| Trainable values | one mask logit for each of 32 × 32 attention heads |
| Intervention | head output immediately before `o_proj` |
| Mask | hard straight-through Gumbel-Sigmoid, temperature 1.0 |
| Mask initialization | Normal(0, 0.01) |
| Objective | CE over [gold, distractor] minus mean keep probability |
| Optimizer | AdamW, learning rate 1.0 |
| Epochs | 500 |
| Batch size | 16, matching the released command default |
| Train order | fixed; no epoch-level shuffle |
| Epoch selection | released code's final dev batch gap + kept-head count × 0.1 |
| Main data protocol | `official-code` |

The public `evaluate()` function appends its logit differences after the batch
loop, so only the final development batch contributes to the checkpoint score.
With 40 development prompts and batch size 16, that means 8 examples. v0.4
reproduces this behavior only for checkpoint selection and simultaneously
records the correct 40-example mean as `dev_full`. Final test metrics always use
all 276 examples.

For `country_capital_city`, the official-code protocol must produce:

```text
train: 360
dev:    40
test:   276
```

It does so by reproducing the released builder: reserve 10% of relation samples
as test queries; pair queries against all relation facts as possible contexts;
retain pairs whose object differs; cap each pair list at 100; expand all 2 × 2
template combinations; and reserve 10% of the expanded train/dev prompts for
development.

The paper describes this more simply as an 80/10/10 split. Because that sentence
and the released builder are not identical, the published-result comparison uses
`official-code`. The earlier subject-disjoint interpretation remains available
as `--head-data-protocol disjoint-80-10-10`, but it is a sensitivity analysis,
not the primary reproduction.

## Install

```bash
python -m pip install -e '.[dev]'
bash scripts/fetch_lre_data.sh
```

Llama-3.1-8B is gated. Accept Meta's access terms and authenticate with Hugging
Face before running a GPU job. Never put a token in a script or repository.

## Verify data construction without a GPU

```bash
python scripts/verify_head_protocol.py
```

The command must print `"passed": true` and `360/40/276`.

## Run head search

First run the full-data, one-epoch compatibility check:

```bash
sbatch slurm/03_head_search_timing.sbatch
```

Only after that succeeds, submit the 500-epoch seed-0 run:

```bash
sbatch slurm/03_head_search.sbatch
```

After the seed-0 run, the optional four-job array completes seeds 0–4:

```bash
sbatch slurm/04_head_search_five_seeds.sbatch
```

After completion, compare the primary run to Table 2 and summarize Appendix-D
overlap with:

```bash
python scripts/compare_table2.py
python scripts/summarize_head_runs.py \
  outputs/head_search_capital_v040_seed*/best_mask.json
```

The included Slurm files are configured for the current A5000 server path,
`batch_ugrad`, and `ariel-v8`. The repeat array uses `%1`, so it requests only
one GPU at a time. Change only the node directive if allocation
requires a different node.

## Outputs

Each head-search directory contains:

- `data_manifest.json`: protocol and exact train/dev/test counts;
- `train_examples.jsonl`, `dev_examples.jsonl`, `test_examples.jsonl`;
- `history.jsonl`: epoch loss, released-code selection dev metrics, corrected
  full-dev metrics, head counts, and selection score;
- `best_mask.json`: the development-selected deterministic mask;
- `test_measurements.jsonl`: all four conditions for every test example;
- `head_search_report.json`: aggregates and paired tests;
- `run_config.json`: arguments, versions, CUDA, GPU, and Slurm job ID.

The test set is evaluated only after the best development checkpoint is fixed.
This follows the paper's statement that test is not used for checkpoint selection,
even though the released code computes test metrics after every epoch.

## Reproduction targets

Table 2 reports, for country–capital, 36 removed heads, a with-context
gold-minus-distractor logit gap of about 7.69 before masking and 13.20 after
masking, and distractor rank changing from about 37.5 to 1289.6. These are
comparison targets, not assertions: exact heads and values vary because mask
learning is stochastic. Appendix D reports 60–96 selected heads across five
runs and pairwise Jaccard overlap from 0.317 to 0.548. A single seed is the first
reproduction run rather than the complete stability study.

## Scope

The implementation uses Hugging Face Llama with an `o_proj` pre-hook instead of
the official repository's modified TransformerLens fork. For Llama this is the
corresponding linear intervention point, but the software stack and numerical
precision can still prevent bit-for-bit equality. Target scoring intentionally
uses the first token after a leading space, matching the released code.

For a correctness-oriented sensitivity analysis, pass
`--checkpoint-selection full-dev`. Do not mix that result with the primary
published-result comparison.

## Citation

```bibtex
@inproceedings{niu-etal-2025-llama,
  title = {Llama See, Llama Do: A Mechanistic Perspective on Contextual
           Entrainment and Distraction in LLMs},
  author = {Niu, Jingcheng and Yuan, Xingdi and Wang, Tong and
            Saghir, Hamidreza and Abdi, Amir H.},
  booktitle = {Proceedings of ACL},
  year = {2025},
  doi = {10.18653/v1/2025.acl-long.791}
}
```
