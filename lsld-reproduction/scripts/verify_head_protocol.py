#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from lsld_repro.data import build_head_search_splits, load_relations, relation_key

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--lre-dir', default='data/lre_dataset')
    parser.add_argument('--relation', default='country_capital_city')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    relation = load_relations(args.lre_dir)[relation_key(args.relation)]
    datasets = build_head_search_splits(relation, max_pairs_per_split=100, seed=args.seed, protocol='official-code')
    counts = {split: len(examples) for split, examples in datasets.items()}
    payload = {'relation': relation.key, 'relation_samples': len(relation.samples), 'context_templates': len(relation.context_templates), 'query_templates': len(relation.query_templates), 'protocol': 'official-code', 'counts': counts, 'passed': counts == {'train': 360, 'dev': 40, 'test': 276}}
    print(json.dumps(payload, indent=2))
    if not payload['passed']:
        raise SystemExit('Unexpected country-capital split counts')
if __name__ == '__main__':
    main()
