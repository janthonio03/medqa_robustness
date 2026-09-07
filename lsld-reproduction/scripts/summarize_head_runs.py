#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path
PAPER_APPENDIX_D_HEAD_COUNTS = [96, 60, 89, 70, 92]
PAPER_APPENDIX_D_JACCARD = [[1.0, 0.381, 0.542, 0.372, 0.516], [0.381, 1.0, 0.419, 0.548, 0.322], [0.542, 0.419, 1.0, 0.5, 0.351], [0.372, 0.548, 0.5, 1.0, 0.317], [0.516, 0.322, 0.351, 0.317, 1.0]]

def removed_heads(path: Path) -> set[tuple[int, int]]:
    with path.open(encoding='utf-8') as handle:
        payload = json.load(handle)
    return {(int(item['layer']), int(item['head'])) for item in payload['removed_heads']}

def jaccard(left: set[tuple[int, int]], right: set[tuple[int, int]]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0

def upper_triangle(matrix: list[list[float]]) -> list[float]:
    return [matrix[row][column] for row in range(len(matrix)) for column in range(row + 1, len(matrix))]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('masks', nargs='+', type=Path)
    args = parser.parse_args()
    heads = [removed_heads(path) for path in args.masks]
    observed_jaccard = [[jaccard(left, right) for right in heads] for left in heads]
    observed_counts = [len(run_heads) for run_heads in heads]
    observed_pairs = upper_triangle(observed_jaccard)
    paper_pairs = upper_triangle(PAPER_APPENDIX_D_JACCARD)
    payload = {'runs': [{'mask': str(path), 'removed_head_count': len(run_heads)} for path, run_heads in zip(args.masks, heads, strict=True)], 'jaccard': observed_jaccard, 'observed_summary': {'mean_removed_head_count': statistics.mean(observed_counts), 'min_removed_head_count': min(observed_counts), 'max_removed_head_count': max(observed_counts), 'mean_pairwise_jaccard': statistics.mean(observed_pairs) if observed_pairs else None, 'min_pairwise_jaccard': min(observed_pairs) if observed_pairs else None, 'max_pairwise_jaccard': max(observed_pairs) if observed_pairs else None}, 'paper_appendix_d_reference': {'removed_head_counts': PAPER_APPENDIX_D_HEAD_COUNTS, 'mean_removed_head_count': statistics.mean(PAPER_APPENDIX_D_HEAD_COUNTS), 'pairwise_jaccard_range': [min(paper_pairs), max(paper_pairs)], 'mean_pairwise_jaccard': statistics.mean(paper_pairs)}}
    print(json.dumps(payload, indent=2))
if __name__ == '__main__':
    main()
