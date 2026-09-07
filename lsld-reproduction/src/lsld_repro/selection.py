from __future__ import annotations
from typing import Sequence, TypeVar
T = TypeVar('T')

def paper_epoch_selection_score(dev_gold_minus_distractor_logit: float, kept_heads: int, head_count_weight: float=0.1) -> float:
    if kept_heads < 0:
        raise ValueError('kept_heads must be non-negative')
    if head_count_weight < 0:
        raise ValueError('head_count_weight must be non-negative')
    return float(dev_gold_minus_distractor_logit) + kept_heads * head_count_weight

def released_code_last_batch(records: Sequence[T], batch_size: int) -> Sequence[T]:
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if not records:
        raise ValueError('records must not be empty')
    last_batch_size = len(records) % batch_size or min(batch_size, len(records))
    return records[-last_batch_size:]
