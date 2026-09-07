from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable
import numpy as np
from scipy.stats import ttest_rel

def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float('nan')

def summarize_measurements(records: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        groups[str(record['relation']), str(record['setting'])].append(record)
    summaries: list[dict[str, object]] = []
    for (relation, setting), items in sorted(groups.items()):
        dstr_no = [float(x['distractor_logit_no_context']) for x in items]
        dstr_with = [float(x['distractor_logit_with_context']) for x in items]
        gold_no = [float(x['gold_logit_no_context']) for x in items]
        gold_with = [float(x['gold_logit_with_context']) for x in items]
        dstr_prob_no = [float(x['distractor_probability_no_context']) for x in items]
        dstr_prob_with = [float(x['distractor_probability_with_context']) for x in items]
        t_result = ttest_rel(dstr_with, dstr_no) if len(items) > 1 else None
        summary = {'relation': relation, 'setting': setting, 'n': len(items), 'gold_logit_no_context': _mean(gold_no), 'gold_logit_with_context': _mean(gold_with), 'gold_logit_delta': _mean([with_ - no for with_, no in zip(gold_with, gold_no, strict=True)]), 'distractor_logit_no_context': _mean(dstr_no), 'distractor_logit_with_context': _mean(dstr_with), 'distractor_logit_delta': _mean([with_ - no for with_, no in zip(dstr_with, dstr_no, strict=True)]), 'distractor_probability_no_context': _mean(dstr_prob_no), 'distractor_probability_with_context': _mean(dstr_prob_with), 'distractor_relative_probability_delta': _mean([(with_ - no) / no if no > 0 else float('nan') for with_, no in zip(dstr_prob_with, dstr_prob_no, strict=True)]), 'paired_t': float(t_result.statistic) if t_result else float('nan'), 'paired_p': float(t_result.pvalue) if t_result else float('nan')}
        summaries.append(summary)
        counterfactual_items = [x for x in items if 'factual_token_logit_delta' in x]
        if counterfactual_items:
            factual_deltas = [float(x['factual_token_logit_delta']) for x in counterfactual_items]
            counterfactual_deltas = [float(x['distractor_logit_delta']) for x in counterfactual_items]
            comparison = ttest_rel(counterfactual_deltas, factual_deltas)
            summary.update({'factual_token_logit_delta': _mean(factual_deltas), 'counterfactual_minus_factual_logit_delta': _mean([cf - factual for cf, factual in zip(counterfactual_deltas, factual_deltas, strict=True)]), 'counterfactual_vs_factual_paired_t': float(comparison.statistic), 'counterfactual_vs_factual_paired_p': float(comparison.pvalue)})
    return summaries

def write_summary_csv(path: str | Path, summaries: list[dict[str, object]]) -> None:
    if not summaries:
        raise ValueError('No summaries to write')
    fieldnames: list[str] = []
    for summary in summaries:
        for key in summary:
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
