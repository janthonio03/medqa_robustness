from __future__ import annotations
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence
import numpy as np
SELECTED_RELATIONS = ('company_hq', 'country_capital_city', 'country_currency', 'country_language', 'country_largest_city', 'food_from_country', 'fruit_inside_color', 'fruit_outside_color', 'landmark_in_country', 'landmark_on_continent', 'product_by_company', 'star_constellation_name', 'task_done_by_tool', 'task_person_type', 'work_location')

def relation_key(name: str) -> str:
    return name.strip().lower().replace('–', '_').replace('-', '_').replace(' ', '_')

@dataclass(frozen=True)
class Relation:
    key: str
    name: str
    relation_type: str
    domain: str
    range: str
    context_templates: tuple[str, ...]
    query_templates: tuple[str, ...]
    samples: tuple[dict[str, str], ...]

@dataclass(frozen=True)
class PromptExample:
    example_id: str
    relation: str
    setting: str
    no_context_prompt: str
    with_context_prompt: str
    gold_text: str
    distractor_text: str
    context_text: str
    factual_context_text: str | None = None
    factual_with_context_prompt: str | None = None
    split: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

def _sentence(text: str) -> str:
    text = text.strip()
    if text.endswith(('.', '!', '?')):
        return text
    return text + '.'

def _render_fact(template: str, subject: str, object_: str) -> str:
    return _sentence(f'{template.format(subject).rstrip()} {object_}')

def _render_query(template: str, subject: str) -> str:
    return template.format(subject).strip()

def load_relations(root: str | Path) -> dict[str, Relation]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f'LRE directory does not exist: {root}. Run scripts/fetch_lre_data.sh first.')
    relations: dict[str, Relation] = {}
    for path in sorted(root.rglob('*.json')):
        with path.open(encoding='utf-8') as handle:
            raw = json.load(handle)
        key = relation_key(raw['name'])
        properties = raw['properties']
        relations[key] = Relation(key=key, name=raw['name'], relation_type=properties['relation_type'], domain=properties['domain_name'], range=properties['range_name'], context_templates=tuple(raw['prompt_templates']), query_templates=tuple(raw['prompt_templates_zs']), samples=tuple(raw['samples']))
    return relations

def resolve_relations(available: dict[str, Relation], requested: Sequence[str] | None) -> list[Relation]:
    keys = [relation_key(x) for x in requested] if requested else list(SELECTED_RELATIONS)
    missing = [key for key in keys if key not in available]
    if missing:
        raise KeyError(f"Relations not found in LRE directory: {', '.join(missing)}")
    return [available[key] for key in keys]

def _valid_pair_indices(samples: Sequence[dict[str, str]]) -> list[tuple[int, int]]:
    return [(query_idx, context_idx) for query_idx, query in enumerate(samples) for context_idx, context in enumerate(samples) if query_idx != context_idx and query['subject'] != context['subject'] and (query['object'] != context['object'])]

def _sample_pairs(samples: Sequence[dict[str, str]], limit: int, rng: random.Random) -> list[tuple[dict[str, str], dict[str, str]]]:
    pairs = _valid_pair_indices(samples)
    if len(pairs) > limit:
        pairs = rng.sample(pairs, limit)
    return [(samples[q], samples[c]) for q, c in pairs]

def _append_example(output: list[PromptExample], *, relation: Relation, setting: str, query: str, context: str, gold: str, distractor: str, factual_context: str | None=None, factual_context_sentence: str | None=None, split: str | None=None) -> None:
    idx = len(output)
    output.append(PromptExample(example_id=f"{relation.key}:{setting}:{split or 'all'}:{idx:07d}", relation=relation.key, setting=setting, no_context_prompt=query, with_context_prompt=f'{context.strip()} {query.strip()}', gold_text=gold, distractor_text=distractor, context_text=context, factual_context_text=factual_context, factual_with_context_prompt=f'{factual_context_sentence.strip()} {query.strip()}' if factual_context_sentence is not None else None, split=split))

def build_related_examples(relation: Relation, *, max_examples: int, rng: random.Random, split: str | None=None, samples: Sequence[dict[str, str]] | None=None) -> list[PromptExample]:
    samples = tuple(samples or relation.samples)
    template_count = len(relation.context_templates) * len(relation.query_templates)
    pair_limit = max(1, (max_examples + template_count - 1) // template_count)
    pairs = _sample_pairs(samples, pair_limit, rng)
    output: list[PromptExample] = []
    for query_sample, context_sample in pairs:
        for context_template in relation.context_templates:
            for query_template in relation.query_templates:
                context = _render_fact(context_template, context_sample['subject'], context_sample['object'])
                query = _render_query(query_template, query_sample['subject'])
                _append_example(output, relation=relation, setting='related', query=query, context=context, gold=query_sample['object'], distractor=context_sample['object'], split=split)
                if len(output) >= max_examples:
                    return output
    return output

def build_counterfactual_examples(relation: Relation, *, max_examples: int, rng: random.Random) -> list[PromptExample]:
    pairs = _sample_pairs(relation.samples, max_examples, rng)
    all_targets = sorted({sample['object'] for sample in relation.samples})
    output: list[PromptExample] = []
    for query_sample, context_sample in pairs:
        candidates = [target for target in all_targets if target not in {query_sample['object'], context_sample['object']}]
        if not candidates:
            continue
        counterfactual = rng.choice(candidates)
        context_template = rng.choice(relation.context_templates)
        query_template = rng.choice(relation.query_templates)
        context = _render_fact(context_template, context_sample['subject'], counterfactual)
        factual_context_sentence = _render_fact(context_template, context_sample['subject'], context_sample['object'])
        query = _render_query(query_template, query_sample['subject'])
        _append_example(output, relation=relation, setting='counterfactual', query=query, context=context, gold=query_sample['object'], distractor=counterfactual, factual_context=context_sample['object'], factual_context_sentence=factual_context_sentence)
        if len(output) >= max_examples:
            break
    return output

def build_irrelevant_examples(relation: Relation, all_relations: Iterable[Relation], *, max_examples: int, rng: random.Random) -> list[PromptExample]:
    query_signature = {relation.domain, relation.range}
    candidates = [other for other in all_relations if other.key != relation.key and query_signature.isdisjoint({other.domain, other.range})]
    if not candidates:
        raise ValueError(f'No domain/range-disjoint relation found for {relation.key}')
    output: list[PromptExample] = []
    attempts = 0
    max_attempts = max_examples * 20
    seen: set[tuple[str, str]] = set()
    while len(output) < max_examples and attempts < max_attempts:
        attempts += 1
        query_sample = rng.choice(relation.samples)
        context_relation = rng.choice(candidates)
        context_sample = rng.choice(context_relation.samples)
        context = _render_fact(rng.choice(context_relation.context_templates), context_sample['subject'], context_sample['object'])
        query = _render_query(rng.choice(relation.query_templates), query_sample['subject'])
        signature = (context, query)
        if signature in seen:
            continue
        seen.add(signature)
        _append_example(output, relation=relation, setting='irrelevant', query=query, context=context, gold=query_sample['object'], distractor=context_sample['object'])
    return output

def brown_random_words(count: int, seed: int) -> list[str]:
    try:
        from nltk.corpus import brown
        words = [word for word in brown.words() if word.isalpha() and 4 <= len(word) <= 14]
    except LookupError as exc:
        raise RuntimeError('NLTK Brown corpus is missing. Run: python -m nltk.downloader brown') from exc
    unique = sorted(set(words), key=str.lower)
    rng = random.Random(seed)
    rng.shuffle(unique)
    if len(unique) < count:
        raise ValueError(f'Brown corpus contains only {len(unique)} eligible unique words')
    return unique[:count]

def build_random_examples(relation: Relation, random_words: Sequence[str], *, max_examples: int, rng: random.Random) -> list[PromptExample]:
    output: list[PromptExample] = []
    seen: set[tuple[str, str]] = set()
    attempts = 0
    max_attempts = max_examples * 20
    while len(output) < max_examples and attempts < max_attempts:
        attempts += 1
        query_sample = rng.choice(relation.samples)
        word = rng.choice(random_words)
        query = _render_query(rng.choice(relation.query_templates), query_sample['subject'])
        signature = (word, query)
        if signature in seen:
            continue
        seen.add(signature)
        _append_example(output, relation=relation, setting='random', query=query, context=word, gold=query_sample['object'], distractor=word)
    return output

def generate_baseline_examples(relations: Sequence[Relation], *, context_relations: Sequence[Relation] | None=None, settings: Sequence[str], max_examples: int, seed: int, random_word_count: int=100) -> list[PromptExample]:
    allowed = {'related', 'irrelevant', 'random', 'counterfactual'}
    unknown = set(settings) - allowed
    if unknown:
        raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
    rng = random.Random(seed)
    random_words = brown_random_words(random_word_count, seed) if 'random' in settings else []
    output: list[PromptExample] = []
    context_relations = list(context_relations or relations)
    for relation in relations:
        if 'related' in settings:
            output.extend(build_related_examples(relation, max_examples=max_examples, rng=rng))
        if 'irrelevant' in settings:
            output.extend(build_irrelevant_examples(relation, context_relations, max_examples=max_examples, rng=rng))
        if 'random' in settings:
            output.extend(build_random_examples(relation, random_words, max_examples=max_examples, rng=rng))
        if 'counterfactual' in settings:
            output.extend(build_counterfactual_examples(relation, max_examples=max_examples, rng=rng))
    return output

def split_samples(samples: Sequence[dict[str, str]], seed: int) -> dict[str, tuple[dict[str, str], ...]]:
    if len(samples) < 10:
        raise ValueError('At least 10 relation samples are required for an 80/10/10 split')
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    n_total = len(shuffled)
    n_test = max(1, round(n_total * 0.1))
    n_dev = max(1, round(n_total * 0.1))
    n_train = n_total - n_dev - n_test
    return {'train': tuple(shuffled[:n_train]), 'dev': tuple(shuffled[n_train:n_train + n_dev]), 'test': tuple(shuffled[n_train + n_dev:])}

def _sklearn_style_split(items: Sequence[object], *, test_fraction: float, rng: np.random.RandomState) -> tuple[list[object], list[object]]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError('test_fraction must be between zero and one')
    if len(items) < 2:
        raise ValueError('At least two items are required for a split')
    n_test = int(math.ceil(len(items) * test_fraction))
    permutation = rng.permutation(len(items)).tolist()
    test_indices = permutation[:n_test]
    train_indices = permutation[n_test:]
    return ([items[index] for index in train_indices], [items[index] for index in test_indices])

def _build_released_head_examples(relation: Relation, *, query_samples: Sequence[dict[str, str]], max_pairs: int, rng: random.Random, split: str) -> list[PromptExample]:
    pairs = [(query_sample, context_sample) for query_sample in query_samples for context_sample in relation.samples if query_sample['object'] != context_sample['object']]
    if len(pairs) > max_pairs:
        pairs = rng.sample(pairs, max_pairs)
    output: list[PromptExample] = []
    for query_template in relation.query_templates:
        for context_template in relation.context_templates:
            for query_sample, context_sample in pairs:
                context = _render_fact(context_template, context_sample['subject'], context_sample['object'])
                query = _render_query(query_template, query_sample['subject'])
                _append_example(output, relation=relation, setting='related', query=query, context=context, gold=query_sample['object'], distractor=context_sample['object'], split=split)
    return output

def build_released_head_search_splits(relation: Relation, *, max_pairs_per_split: int, seed: int) -> dict[str, list[PromptExample]]:
    split_rng = np.random.RandomState(seed)
    train_dev_queries_raw, test_queries_raw = _sklearn_style_split(relation.samples, test_fraction=0.1, rng=split_rng)
    train_dev_queries = [dict(item) for item in train_dev_queries_raw]
    test_queries = [dict(item) for item in test_queries_raw]
    pair_rng = random.Random(seed)
    train_dev_examples = _build_released_head_examples(relation, query_samples=train_dev_queries, max_pairs=max_pairs_per_split, rng=pair_rng, split='train_dev')
    test_examples = _build_released_head_examples(relation, query_samples=test_queries, max_pairs=max_pairs_per_split, rng=pair_rng, split='test')
    train_raw, dev_raw = _sklearn_style_split(train_dev_examples, test_fraction=0.1, rng=split_rng)

    def relabel(items: Sequence[object], split: str) -> list[PromptExample]:
        relabeled: list[PromptExample] = []
        for index, raw in enumerate(items):
            if not isinstance(raw, PromptExample):
                raise TypeError('Expected PromptExample after head-search split')
            payload = raw.to_dict()
            payload['split'] = split
            payload['example_id'] = f'{relation.key}:related:{split}:{index:07d}'
            relabeled.append(PromptExample(**payload))
        return relabeled
    return {'train': relabel(train_raw, 'train'), 'dev': relabel(dev_raw, 'dev'), 'test': relabel(test_examples, 'test')}

def build_head_search_splits(relation: Relation, *, max_pairs_per_split: int, seed: int, protocol: str='official-code') -> dict[str, list[PromptExample]]:
    if protocol == 'official-code':
        return build_released_head_search_splits(relation, max_pairs_per_split=max_pairs_per_split, seed=seed)
    if protocol != 'disjoint-80-10-10':
        raise ValueError(f'Unknown head-search data protocol: {protocol}')
    splits = split_samples(relation.samples, seed)
    output: dict[str, list[PromptExample]] = {}
    template_count = len(relation.context_templates) * len(relation.query_templates)
    max_examples = max_pairs_per_split * template_count
    for offset, (split, samples) in enumerate(splits.items()):
        output[split] = build_related_examples(relation, max_examples=max_examples, rng=random.Random(seed + offset), split=split, samples=samples)
    return output
