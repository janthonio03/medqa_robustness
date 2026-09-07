# Paper and released-code correspondence

Audited sources:

- ACL 2025 paper: <https://aclanthology.org/2025.acl-long.791/>
- official repository: <https://github.com/frankniujc/entrainment>
- official commit: `c22d893dd358b27c057ef6ba37c2da1101866497`

## Head-search settings

| Item | Paper | Released code | v0.4 primary run |
|---|---|---|---|
| Model | Llama-3.1-8B | Llama-3.1-8B | same |
| Epochs | 500 | 500 | same |
| Optimizer | AdamW | AdamW | same |
| Learning rate | 1.0 | 1.0 | same |
| Sparsity coefficient | 1.0 | 1.0 | same |
| Gumbel temperature | 1.0 | 1.0 | same |
| Batch size | not stated | CLI default 16 | 16 |
| Train shuffle | not stated | false | false |
| Selection | dev gap + kept heads × 0.1 | final dev batch only due to loop placement | same compatibility behavior |
| Test during search | not used for selection | evaluated every epoch | evaluated once after selection |

## Data split discrepancy

The paper says 80/10/10. The released `get_headsearch_data` function instead:

1. uses `train_test_split(samples, test_size=0.1)` for query facts;
2. pairs those queries with all facts in the relation as contexts;
3. keeps pairs where query object and context object differ;
4. samples at most 100 pairs;
5. expands every query/context template combination;
6. splits the expanded train/dev prompts again with `test_size=0.1`.

With the official 24-item country-capital relation and 2 × 2 templates, the
result is exactly 360 train, 40 dev, and 276 test prompts. Version 0.3 makes this
`official-code` protocol the default because it is the operational definition
behind the released result.

The original repository leaves `random_state` implicit. It seeds NumPy and
Python globally and lets the two NumPy splits consume one continued RNG stream.
v0.4 uses local deterministic RNG objects derived from `--seed` while preserving
that continued state, the released counts, pairing rule, cap, and template
order.

`disjoint-80-10-10` preserves version 0.2's stricter subject-disjoint
interpretation for sensitivity analysis. It should not be mixed with the main
published-result comparison.

## Mask and intervention

The released modified TransformerLens model multiplies each head's `z` by a
hard straight-through Gumbel-Sigmoid mask before the head output matrices.
v0.4 applies the same per-head mask to the concatenated head tensor immediately
before Hugging Face Llama's `self_attn.o_proj`. For Llama, the two operations are
the corresponding intervention point.

Mask logits are float32 and initialized from Normal(0, 0.01). A deterministic
mask keeps values greater than zero. Gumbel sampling uses the released operand
order and epsilon placement.

## Objective and checkpoint selection

The released optimization loss is:

```text
cross_entropy([gold_logit, distractor_logit], gold)
- 1.0 * mean(sigmoid(mask_logits))
```

The second term rewards keeping heads active. The development checkpoint score
is the expression in paper footnote 5:

```text
dev gold-minus-distractor logit gap + kept-head count * 0.1
```

For 1,024 total heads, this is equivalent up to the constant 102.4 to
`dev gap - removed-head count * 0.1`.

## Checkpoint-selection compatibility and corrected reporting

The released `evaluate` implementation appends logit/probability differences
outside its batch loop, so checkpoint selection sees only the final dev batch.
The published checkpoints and Appendix-D counts are downstream of this
behavior. Therefore the v0.4 paper-first default,
`--checkpoint-selection released-code-last-batch`, uses the final 8 of 40 dev
examples for the selection score. It also records a corrected 40-example mean
as `dev_full`. `--checkpoint-selection full-dev` retains the v0.3
correctness-oriented behavior as a sensitivity analysis.

Final test evaluation never reproduces the aggregation bug. The report uses all
276 test examples and adds auditable
per-example rows plus paired tests for:

- distractor rank after masking versus all heads, with context;
- gold-minus-distractor logit gap after masking versus all heads, with context;
- masked versus all-head contextual distractor-logit effect.

## Published comparison targets

Table 2's representative country-capital result removes 36 heads, changes the
with-context gold-minus-distractor gap from 7.69 to 13.20, and changes the
distractor rank from 37.5 to 1289.6. Appendix D is a separate five-run stability
experiment: its runs select 96, 60, 89, 70, and 92 heads, with pairwise Jaccard
overlap from 0.317 to 0.548 (mean 0.4268). The paper's printed density beside
the 89-head row is internally inconsistent with 89 / 1024; v0.4 uses the
reported head count rather than silently changing it.

## Baseline corpus discrepancy

The paper says random words are drawn from Brown. The public code currently
references NLTK ABC. The baseline command follows the paper and records Brown
corpus use. This does not affect the related-context head-search experiment.

## Target tokenization

The released code scores the first token after prepending a space. v0.4 matches
that behavior and stores the full target string and complete tokenization so
multi-token objects such as `New Delhi` remain visible as a limitation.

## Remaining non-identities

- Hugging Face replaces the official modified TransformerLens stack.
- The paper does not specify numeric dtype; the A5000 run records the resolved
  dtype and package versions in `run_config.json`.
- The released environment is not version-pinned, so bit-for-bit reproduction
  is not guaranteed.
- The paper reports five stochastic reruns. The seed-0 run is the primary check;
  use the included array job for seeds 0–4 before making a stability claim.
