from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from .data import build_head_search_splits, generate_baseline_examples, load_relations, relation_key, resolve_relations
from .experiments import evaluate_mask_conditions, measure_examples, run_baseline, train_head_mask
from .io_utils import environment_metadata, prepare_output_dir, set_seed, write_json, write_jsonl
from .masking import AttentionHeadMask
from .modeling import load_model_and_tokenizer

def add_model_args(parser: argparse.ArgumentParser, *, default_batch_size: int=1) -> None:
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', choices=['auto', 'bfloat16', 'float16', 'float32'], default='auto')
    parser.add_argument('--batch-size', type=int, default=default_batch_size)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, required=True)

def add_lre_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--lre-dir', type=Path, required=True)

def load_runtime(args):
    set_seed(args.seed)
    output_dir = prepare_output_dir(args.output_dir)
    model, tokenizer = load_model_and_tokenizer(args.model, dtype=args.dtype, device=args.device)
    return (output_dir, model, tokenizer)

def save_run_config(args, output_dir: Path) -> None:
    payload = {'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != 'handler'}, 'environment': environment_metadata()}
    write_json(output_dir / 'run_config.json', payload)

def command_smoke(args) -> None:
    output_dir, model, tokenizer = load_runtime(args)
    query = 'Greece is located on the continent of'
    contexts = ['Iraq is located on the continent of Asia.', 'Asia is the largest continent in the world by both land area and population.', 'Asia.']
    from .data import PromptExample
    examples = [PromptExample(example_id=f'smoke:{idx}', relation='country_capital_city', setting='smoke', no_context_prompt=query, with_context_prompt=f'{context} {query}', gold_text='Europe', distractor_text='Asia', context_text=context) for idx, context in enumerate(contexts)]
    measurements = measure_examples(model, tokenizer, examples, batch_size=args.batch_size)
    write_jsonl(output_dir / 'smoke_measurements.jsonl', measurements)
    passed = all((float(x['distractor_logit_delta']) > 0 for x in measurements))
    write_json(output_dir / 'smoke_result.json', {'passed': passed, 'criterion': 'Asia logit delta > 0 for all contexts'})
    save_run_config(args, output_dir)
    print(json.dumps({'passed': passed, 'output_dir': str(output_dir)}, indent=2))

def command_baseline(args) -> None:
    available = load_relations(args.lre_dir)
    relations = resolve_relations(available, args.relations)
    context_relations = resolve_relations(available, None)
    examples = generate_baseline_examples(relations, context_relations=context_relations, settings=args.settings, max_examples=args.max_examples, seed=args.seed, random_word_count=args.random_word_count)
    output_dir, model, tokenizer = load_runtime(args)
    summaries = run_baseline(model, tokenizer, examples, batch_size=args.batch_size, output_dir=output_dir)
    save_run_config(args, output_dir)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))

def _head_datasets(args):
    available = load_relations(args.lre_dir)
    key = relation_key(args.relation)
    if key not in available:
        raise KeyError(f'Relation not found: {key}')
    return build_head_search_splits(available[key], max_pairs_per_split=args.max_pairs_per_split, seed=args.seed, protocol=args.head_data_protocol)

def _write_head_data_manifest(args, datasets, output_dir: Path) -> None:
    counts = {split: len(examples) for split, examples in datasets.items()}
    payload = {'implementation_version': '0.4.0', 'official_repository_commit': 'c22d893dd358b27c057ef6ba37c2da1101866497', 'relation': relation_key(args.relation), 'head_data_protocol': args.head_data_protocol, 'seed': args.seed, 'max_pairs_per_split': args.max_pairs_per_split, 'data_rng_protocol': 'continued NumPy RandomState across both released-code splits; continued Python Random stream across pair sampling', 'counts': counts, 'official_country_capital_expected_counts': {'train': 360, 'dev': 40, 'test': 276}, 'note': "official-code follows the released repository's 90/10 query split, all-relation context pool, 100-pair cap, template expansion, and prompt-level 90/10 train/dev split, including its continued RNG stream. The paper text separately describes the aggregate split as 80/10/10."}
    write_json(output_dir / 'data_manifest.json', payload)
    if payload['relation'] == 'country_capital_city' and args.head_data_protocol == 'official-code' and (args.max_pairs_per_split == 100) and (counts != payload['official_country_capital_expected_counts']):
        raise RuntimeError(f'Official country-capital data counts do not match 360/40/276: {counts}')

def command_head_search(args) -> None:
    datasets = _head_datasets(args)
    output_dir, model, tokenizer = load_runtime(args)
    for split, examples in datasets.items():
        write_jsonl(output_dir / f'{split}_examples.jsonl', examples)
    _write_head_data_manifest(args, datasets, output_dir)
    save_run_config(args, output_dir)
    controller = AttentionHeadMask(model, temperature=args.temperature)
    try:
        report = train_head_mask(model, tokenizer, controller, datasets, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, sparsity_lambda=args.sparsity_lambda, head_count_weight=args.head_count_weight, checkpoint_selection=args.checkpoint_selection, output_dir=output_dir)
    finally:
        controller.close()
    print(json.dumps(report['test'], indent=2))

def command_evaluate_mask(args) -> None:
    datasets = _head_datasets(args)
    output_dir, model, tokenizer = load_runtime(args)
    _write_head_data_manifest(args, datasets, output_dir)
    save_run_config(args, output_dir)
    with args.mask.open(encoding='utf-8') as handle:
        mask_payload = json.load(handle)
    mask_values = mask_payload.get('keep_mask', mask_payload.get('mask'))
    if mask_values is None:
        raise KeyError('Mask JSON must contain `keep_mask` or `mask`')
    mask = torch.tensor(mask_values, dtype=torch.float32, device=next(model.parameters()).device)
    controller = AttentionHeadMask(model, temperature=1.0)
    try:
        report = evaluate_mask_conditions(model, tokenizer, controller, datasets['test'], batch_size=args.batch_size, mask=mask, output_dir=output_dir)
    finally:
        controller.close()
    write_json(output_dir / 'mask_evaluation.json', report)
    print(json.dumps(report, indent=2))

def add_head_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--max-pairs-per-split', type=int, default=100)
    parser.add_argument('--head-data-protocol', choices=['official-code', 'disjoint-80-10-10'], default='official-code', help="official-code reproduces the released data builder and is the default for published-result comparison; disjoint-80-10-10 is a separate sensitivity analysis of the paper's split description")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='lsld', description='Llama See, Llama Do reproduction')
    subparsers = parser.add_subparsers(dest='command', required=True)
    smoke = subparsers.add_parser('smoke', help='Run Figure-1-style examples')
    add_model_args(smoke)
    smoke.set_defaults(handler=command_smoke)
    baseline = subparsers.add_parser('baseline', help='Measure paired contextual-entrainment shifts')
    add_model_args(baseline)
    add_lre_args(baseline)
    baseline.add_argument('--relations', nargs='+')
    baseline.add_argument('--settings', nargs='+', default=['related', 'irrelevant', 'random', 'counterfactual'], choices=['related', 'irrelevant', 'random', 'counterfactual'])
    baseline.add_argument('--max-examples', type=int, default=1000, help='Maximum examples per relation and setting')
    baseline.add_argument('--random-word-count', type=int, default=100)
    baseline.set_defaults(handler=command_baseline)
    head_search = subparsers.add_parser('head-search', help='Learn a sparse causal attention-head ablation mask')
    add_model_args(head_search, default_batch_size=16)
    add_lre_args(head_search)
    head_search.add_argument('--relation', required=True)
    head_search.add_argument('--epochs', type=int, default=500)
    add_head_data_args(head_search)
    head_search.add_argument('--learning-rate', type=float, default=1.0)
    head_search.add_argument('--sparsity-lambda', type=float, default=1.0)
    head_search.add_argument('--temperature', type=float, default=1.0)
    head_search.add_argument('--head-count-weight', type=float, default=0.1, help="Weight for the paper's epoch-selection rule: dev logit difference + kept head count * weight (paper value: 0.1)")
    head_search.add_argument('--checkpoint-selection', choices=['released-code-last-batch', 'full-dev'], default='released-code-last-batch', help='released-code-last-batch reproduces the public evaluate() behavior used by checkpoint selection; full-dev uses the correct mean over every development example')
    head_search.set_defaults(handler=command_head_search)
    evaluate = subparsers.add_parser('evaluate-mask', help='Evaluate a previously saved deterministic mask')
    add_model_args(evaluate, default_batch_size=16)
    add_lre_args(evaluate)
    evaluate.add_argument('--relation', required=True)
    evaluate.add_argument('--mask', type=Path, required=True)
    add_head_data_args(evaluate)
    evaluate.set_defaults(handler=command_evaluate_mask)
    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)
if __name__ == '__main__':
    main()
