#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
REFERENCE = {'removed_heads': 36.0, 'all_heads_no_context.gold_minus_distractor_logit': 10.76, 'all_heads_with_context.gold_minus_distractor_logit': 7.69, 'masked_no_context.gold_minus_distractor_logit': 11.62, 'masked_with_context.gold_minus_distractor_logit': 13.2, 'all_heads_no_context.distractor_rank': 1756.7, 'all_heads_with_context.distractor_rank': 37.5, 'masked_no_context.distractor_rank': 1707.3, 'masked_with_context.distractor_rank': 1289.6}

def nested(payload: dict[str, object], dotted_key: str) -> float:
    current: object = payload
    for key in dotted_key.split('.'):
        if not isinstance(current, dict):
            raise KeyError(dotted_key)
        current = current[key]
    return float(current)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('report', nargs='?', default='outputs/head_search_capital_v040_seed0/head_search_report.json')
    args = parser.parse_args()
    with open(args.report, encoding='utf-8') as handle:
        report = json.load(handle)
    test = report['test']
    rows = []
    for metric, reference in REFERENCE.items():
        observed = nested(test, metric)
        rows.append({'metric': metric, 'paper_table_2': reference, 'observed': observed, 'observed_minus_paper': observed - reference})
    print(json.dumps(rows, indent=2))
    print(json.dumps({'paired_rank_test': test.get('paired_tests', {}).get('distractor_rank_masked_minus_all_heads_with_context'), 'paper_rank_p_upper_bound': 6.9e-54}, indent=2))
if __name__ == '__main__':
    main()
