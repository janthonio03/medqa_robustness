import json
import random
from lsld_repro.data import Relation, build_head_search_splits, build_counterfactual_examples, build_irrelevant_examples, build_related_examples, relation_key, split_samples

def make_relation(name='country capital city', domain='country', range_='city', *, sample_count=12, context_template_count=1, query_template_count=1):
    return Relation(key=relation_key(name), name=name, relation_type='factual', domain=domain, range=range_, context_templates=tuple((f'Context template {idx} for {{}} is' for idx in range(context_template_count))), query_templates=tuple((f'Query template {idx} for {{}} is' for idx in range(query_template_count))), samples=tuple(({'subject': f'subject-{idx}', 'object': f'target-{idx}'} for idx in range(sample_count))))

def test_related_context_uses_different_subject_and_target():
    relation = make_relation()
    examples = build_related_examples(relation, max_examples=8, rng=random.Random(0))
    assert len(examples) == 8
    for example in examples:
        assert example.setting == 'related'
        assert example.gold_text != example.distractor_text
        assert example.context_text in example.with_context_prompt
        assert example.no_context_prompt in example.with_context_prompt

def test_counterfactual_tracks_factual_and_replacement_targets():
    relation = make_relation()
    examples = build_counterfactual_examples(relation, max_examples=6, rng=random.Random(1))
    assert len(examples) == 6
    for example in examples:
        assert example.factual_context_text is not None
        assert example.distractor_text != example.factual_context_text
        assert example.distractor_text != example.gold_text

def test_irrelevant_relation_has_disjoint_domain_and_range():
    query_relation = make_relation()
    context_relation = make_relation('fruit inside color', 'fruit', 'color')
    examples = build_irrelevant_examples(query_relation, [query_relation, context_relation], max_examples=5, rng=random.Random(2))
    assert len(examples) == 5
    assert all((example.setting == 'irrelevant' for example in examples))
    assert all(('subject-' in example.context_text for example in examples))

def test_split_is_disjoint_and_complete():
    relation = make_relation()
    splits = split_samples(relation.samples, seed=7)
    subjects = {split: {sample['subject'] for sample in samples} for split, samples in splits.items()}
    assert len(subjects['train']) == 10
    assert len(subjects['dev']) == 1
    assert len(subjects['test']) == 1
    assert subjects['train'].isdisjoint(subjects['dev'])
    assert subjects['train'].isdisjoint(subjects['test'])
    assert subjects['dev'].isdisjoint(subjects['test'])

def test_official_code_head_search_protocol_matches_released_counts():
    relation = make_relation(sample_count=24, context_template_count=2, query_template_count=2)
    first = build_head_search_splits(relation, max_pairs_per_split=100, seed=0, protocol='official-code')
    second = build_head_search_splits(relation, max_pairs_per_split=100, seed=0, protocol='official-code')
    assert {split: len(items) for split, items in first.items()} == {'train': 360, 'dev': 40, 'test': 276}
    assert first == second
    assert first['dev'][0].no_context_prompt == 'Query template 1 for subject-20 is'
    assert first['dev'][0].context_text == 'Context template 0 for subject-9 is target-9.'
    assert all((example.gold_text != example.distractor_text for examples in first.values() for example in examples))
    assert all((example.split == split for split, examples in first.items() for example in examples))
