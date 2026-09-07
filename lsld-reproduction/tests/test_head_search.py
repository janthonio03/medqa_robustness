import pytest
from lsld_repro.selection import paper_epoch_selection_score, released_code_last_batch

def test_paper_epoch_score_matches_equivalent_removed_head_form():
    dev_gap = 5.091796875
    kept_heads = 562
    removed_heads = 1024 - kept_heads
    published_form = paper_epoch_selection_score(dev_gap, kept_heads, 0.1)
    centered_form = dev_gap - removed_heads * 0.1
    assert published_form == 61.291796875
    assert abs(published_form - 102.4 - centered_form) < 1e-12

def test_released_code_selection_uses_only_partial_last_batch():
    assert list(released_code_last_batch(list(range(40)), 16)) == list(range(32, 40))

def test_released_code_selection_uses_full_last_batch_when_evenly_divisible():
    assert list(released_code_last_batch(list(range(32)), 16)) == list(range(16, 32))

def test_released_code_selection_rejects_empty_records():
    with pytest.raises(ValueError, match='must not be empty'):
        released_code_last_batch([], 16)
