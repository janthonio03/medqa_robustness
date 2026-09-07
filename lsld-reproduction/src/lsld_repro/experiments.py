from __future__ import annotations
import json
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import ttest_rel
from .data import PromptExample
from .io_utils import batches, write_json, write_jsonl
from .masking import AttentionHeadMask
from .metrics import summarize_measurements, write_summary_csv
from .modeling import first_leading_space_token, gather_target_scores, last_token_logits
from .selection import paper_epoch_selection_score, released_code_last_batch

def _targets_for_examples(tokenizer, examples: Sequence[PromptExample]):
    gold = [first_leading_space_token(tokenizer, item.gold_text) for item in examples]
    distractor = [first_leading_space_token(tokenizer, item.distractor_text) for item in examples]
    return (gold, distractor)

@torch.no_grad()
def measure_examples(model, tokenizer, examples: Sequence[PromptExample], *, batch_size: int) -> list[dict[str, object]]:
    gold_targets, distractor_targets = _targets_for_examples(tokenizer, examples)
    records: list[dict[str, object]] = []
    indexed = list(zip(examples, gold_targets, distractor_targets, strict=True))
    for batch in batches(indexed, batch_size):
        batch_examples = [item[0] for item in batch]
        device = next(model.parameters()).device
        gold_ids = torch.tensor([item[1].token_id for item in batch], device=device)
        distractor_ids = torch.tensor([item[2].token_id for item in batch], device=device)
        no_context_logits = last_token_logits(model, tokenizer, [item.no_context_prompt for item in batch_examples])
        with_context_logits = last_token_logits(model, tokenizer, [item.with_context_prompt for item in batch_examples])
        gold_no = gather_target_scores(no_context_logits, gold_ids)
        gold_with = gather_target_scores(with_context_logits, gold_ids)
        distractor_no = gather_target_scores(no_context_logits, distractor_ids)
        distractor_with = gather_target_scores(with_context_logits, distractor_ids)
        factual_by_row: dict[int, tuple[object, dict[str, torch.Tensor], dict[str, torch.Tensor]]] = {}
        factual_rows = [row for row, example in enumerate(batch_examples) if example.factual_context_text is not None and example.factual_with_context_prompt is not None]
        if factual_rows:
            factual_targets = [first_leading_space_token(tokenizer, batch_examples[row].factual_context_text or '') for row in factual_rows]
            factual_ids = torch.tensor([target.token_id for target in factual_targets], device=device)
            factual_no = gather_target_scores(no_context_logits[factual_rows], factual_ids)
            factual_with_logits = last_token_logits(model, tokenizer, [batch_examples[row].factual_with_context_prompt or '' for row in factual_rows])
            factual_with = gather_target_scores(factual_with_logits, factual_ids)
            for local_row, original_row in enumerate(factual_rows):
                factual_by_row[original_row] = (factual_targets[local_row], {key: value[local_row] for key, value in factual_no.items()}, {key: value[local_row] for key, value in factual_with.items()})
        for row, (example, gold_target, distractor_target) in enumerate(batch):
            record = example.to_dict()
            record.update({'gold_token_id': gold_target.token_id, 'gold_token_text': gold_target.token_text, 'gold_all_token_ids': list(gold_target.all_token_ids), 'distractor_token_id': distractor_target.token_id, 'distractor_token_text': distractor_target.token_text, 'distractor_all_token_ids': list(distractor_target.all_token_ids), 'gold_logit_no_context': float(gold_no['logit'][row].float().cpu()), 'gold_logit_with_context': float(gold_with['logit'][row].float().cpu()), 'gold_probability_no_context': float(gold_no['probability'][row].float().cpu()), 'gold_probability_with_context': float(gold_with['probability'][row].float().cpu()), 'gold_rank_no_context': int(gold_no['rank'][row].cpu()), 'gold_rank_with_context': int(gold_with['rank'][row].cpu()), 'distractor_logit_no_context': float(distractor_no['logit'][row].float().cpu()), 'distractor_logit_with_context': float(distractor_with['logit'][row].float().cpu()), 'distractor_probability_no_context': float(distractor_no['probability'][row].float().cpu()), 'distractor_probability_with_context': float(distractor_with['probability'][row].float().cpu()), 'distractor_rank_no_context': int(distractor_no['rank'][row].cpu()), 'distractor_rank_with_context': int(distractor_with['rank'][row].cpu())})
            record['gold_logit_delta'] = record['gold_logit_with_context'] - record['gold_logit_no_context']
            record['distractor_logit_delta'] = record['distractor_logit_with_context'] - record['distractor_logit_no_context']
            if row in factual_by_row:
                factual_target, factual_no, factual_with = factual_by_row[row]
                factual_delta = float((factual_with['logit'] - factual_no['logit']).float().cpu())
                record.update({'factual_context_token_id': factual_target.token_id, 'factual_context_token_text': factual_target.token_text, 'factual_context_all_token_ids': list(factual_target.all_token_ids), 'factual_token_logit_no_context': float(factual_no['logit'].float().cpu()), 'factual_token_logit_factual_context': float(factual_with['logit'].float().cpu()), 'factual_token_logit_delta': factual_delta, 'counterfactual_minus_factual_logit_delta': float(record['distractor_logit_delta'] - factual_delta)})
            records.append(record)
    return records

def run_baseline(model, tokenizer, examples: Sequence[PromptExample], *, batch_size: int, output_dir: Path) -> list[dict[str, object]]:
    write_jsonl(output_dir / 'examples.jsonl', examples)
    measurements = measure_examples(model, tokenizer, examples, batch_size=batch_size)
    write_jsonl(output_dir / 'measurements.jsonl', measurements)
    summaries = summarize_measurements(measurements)
    write_summary_csv(output_dir / 'summary.csv', summaries)
    return summaries

@torch.no_grad()
def score_prompt_condition(model, tokenizer, examples: Sequence[PromptExample], *, prompt_field: str, batch_size: int) -> list[dict[str, object]]:
    gold_targets, distractor_targets = _targets_for_examples(tokenizer, examples)
    records: list[dict[str, object]] = []
    indexed = list(zip(examples, gold_targets, distractor_targets, strict=True))
    for batch in batches(indexed, batch_size):
        device = next(model.parameters()).device
        prompts = [getattr(item[0], prompt_field) for item in batch]
        gold_ids = torch.tensor([item[1].token_id for item in batch], device=device)
        dstr_ids = torch.tensor([item[2].token_id for item in batch], device=device)
        logits = last_token_logits(model, tokenizer, prompts)
        gold = gather_target_scores(logits, gold_ids)
        dstr = gather_target_scores(logits, dstr_ids)
        for row, (example, gold_target, dstr_target) in enumerate(batch):
            gold_logit = float(gold['logit'][row].float().cpu())
            dstr_logit = float(dstr['logit'][row].float().cpu())
            records.append({'example_id': example.example_id, 'gold_text': example.gold_text, 'distractor_text': example.distractor_text, 'gold_token_id': gold_target.token_id, 'distractor_token_id': dstr_target.token_id, 'gold_logit': gold_logit, 'distractor_logit': dstr_logit, 'gold_minus_distractor_logit': gold_logit - dstr_logit, 'gold_probability': float(gold['probability'][row].float().cpu()), 'distractor_probability': float(dstr['probability'][row].float().cpu()), 'gold_rank': int(gold['rank'][row].cpu()), 'distractor_rank': int(dstr['rank'][row].cpu())})
    return records

def summarize_prompt_scores(records: Sequence[dict[str, object]]) -> dict[str, float | int]:
    fields = ('gold_logit', 'distractor_logit', 'gold_minus_distractor_logit', 'gold_probability', 'distractor_probability', 'gold_rank', 'distractor_rank')
    result: dict[str, float | int] = {'n': len(records)}
    result.update({field: float(np.mean([float(record[field]) for record in records])) for field in fields})
    return result

@torch.no_grad()
def evaluate_prompt_condition(model, tokenizer, examples: Sequence[PromptExample], *, prompt_field: str, batch_size: int) -> dict[str, float | int]:
    records = score_prompt_condition(model, tokenizer, examples, prompt_field=prompt_field, batch_size=batch_size)
    return summarize_prompt_scores(records)

def _paired_result(after: Sequence[float], before: Sequence[float]) -> dict[str, float]:
    result = ttest_rel(after, before) if len(after) > 1 else None
    differences = [a - b for a, b in zip(after, before, strict=True)]
    return {'mean_difference': float(np.mean(differences)), 'paired_t': float(result.statistic) if result else float('nan'), 'paired_p': float(result.pvalue) if result else float('nan')}

def evaluate_mask_conditions(model, tokenizer, controller: AttentionHeadMask, examples: Sequence[PromptExample], *, batch_size: int, mask: torch.Tensor, output_dir: Path | None=None) -> dict[str, object]:
    controller.mode = 'all_on'
    all_on_no_rows = score_prompt_condition(model, tokenizer, examples, prompt_field='no_context_prompt', batch_size=batch_size)
    all_on_with_rows = score_prompt_condition(model, tokenizer, examples, prompt_field='with_context_prompt', batch_size=batch_size)
    controller.set_external_mask(mask)
    masked_no_rows = score_prompt_condition(model, tokenizer, examples, prompt_field='no_context_prompt', batch_size=batch_size)
    masked_with_rows = score_prompt_condition(model, tokenizer, examples, prompt_field='with_context_prompt', batch_size=batch_size)
    condition_rows = {'all_heads_no_context': all_on_no_rows, 'all_heads_with_context': all_on_with_rows, 'masked_no_context': masked_no_rows, 'masked_with_context': masked_with_rows}
    by_condition = {condition: {str(row['example_id']): row for row in rows} for condition, rows in condition_rows.items()}
    detailed: list[dict[str, object]] = []
    for example in examples:
        row: dict[str, object] = {'example_id': example.example_id}
        for condition in condition_rows:
            values = by_condition[condition][example.example_id]
            for key, value in values.items():
                if key not in {'example_id', 'gold_text', 'distractor_text'}:
                    row[f'{condition}.{key}'] = value
        row['gold_text'] = example.gold_text
        row['distractor_text'] = example.distractor_text
        row['all_heads.context_distractor_logit_delta'] = float(row['all_heads_with_context.distractor_logit']) - float(row['all_heads_no_context.distractor_logit'])
        row['masked.context_distractor_logit_delta'] = float(row['masked_with_context.distractor_logit']) - float(row['masked_no_context.distractor_logit'])
        detailed.append(row)
    all_with_dstr_rank = [float(row['all_heads_with_context.distractor_rank']) for row in detailed]
    masked_with_dstr_rank = [float(row['masked_with_context.distractor_rank']) for row in detailed]
    all_with_gap = [float(row['all_heads_with_context.gold_minus_distractor_logit']) for row in detailed]
    masked_with_gap = [float(row['masked_with_context.gold_minus_distractor_logit']) for row in detailed]
    all_context_effect = [float(row['all_heads.context_distractor_logit_delta']) for row in detailed]
    masked_context_effect = [float(row['masked.context_distractor_logit_delta']) for row in detailed]
    removed = controller.removed_heads(mask)
    report = {'total_heads': controller.total_heads, 'kept_heads': controller.total_heads - len(removed), 'removed_heads': len(removed), 'all_heads_no_context': summarize_prompt_scores(all_on_no_rows), 'all_heads_with_context': summarize_prompt_scores(all_on_with_rows), 'masked_no_context': summarize_prompt_scores(masked_no_rows), 'masked_with_context': summarize_prompt_scores(masked_with_rows), 'paired_tests': {'distractor_rank_masked_minus_all_heads_with_context': _paired_result(masked_with_dstr_rank, all_with_dstr_rank), 'gold_minus_distractor_gap_masked_minus_all_heads_with_context': _paired_result(masked_with_gap, all_with_gap), 'context_distractor_effect_masked_minus_all_heads': _paired_result(masked_context_effect, all_context_effect)}}
    if output_dir is not None:
        write_jsonl(output_dir / 'test_measurements.jsonl', detailed)
    return report

def train_head_mask(model, tokenizer, controller: AttentionHeadMask, datasets: dict[str, Sequence[PromptExample]], *, epochs: int, batch_size: int, learning_rate: float, sparsity_lambda: float, head_count_weight: float, checkpoint_selection: str, output_dir: Path) -> dict[str, object]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    optimizer = torch.optim.AdamW([controller.mask_logits], lr=learning_rate)
    history_path = output_dir / 'history.jsonl'
    best_score = -float('inf')
    best_payload: dict[str, object] | None = None
    train_examples = list(datasets['train'])
    dev_examples = list(datasets['dev'])
    test_examples = list(datasets['test'])
    with history_path.open('w', encoding='utf-8') as history:
        for epoch in range(1, epochs + 1):
            controller.mode = 'stochastic'
            losses: list[float] = []
            effects: list[float] = []
            keeps: list[float] = []
            for batch in batches(train_examples, batch_size):
                gold_targets, dstr_targets = _targets_for_examples(tokenizer, batch)
                device = next(model.parameters()).device
                gold_ids = torch.tensor([x.token_id for x in gold_targets], device=device)
                dstr_ids = torch.tensor([x.token_id for x in dstr_targets], device=device)
                logits = last_token_logits(model, tokenizer, [x.with_context_prompt for x in batch])
                gold_logits = logits.gather(1, gold_ids[:, None]).squeeze(1)
                dstr_logits = logits.gather(1, dstr_ids[:, None]).squeeze(1)
                binary_logits = torch.stack([gold_logits, dstr_logits], dim=1)
                effect_loss = F.cross_entropy(binary_logits, torch.zeros(len(batch), dtype=torch.long, device=device))
                keep_probability = controller.keep_probabilities().mean()
                loss = effect_loss - sparsity_lambda * keep_probability
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                effects.append(float(effect_loss.detach().cpu()))
                keeps.append(float(keep_probability.detach().cpu()))
            mask = controller.deterministic_mask().detach()
            controller.set_external_mask(mask)
            dev_rows = score_prompt_condition(model, tokenizer, dev_examples, prompt_field='with_context_prompt', batch_size=batch_size)
            dev_full = summarize_prompt_scores(dev_rows)
            if checkpoint_selection == 'released-code-last-batch':
                selection_rows = released_code_last_batch(dev_rows, batch_size)
            elif checkpoint_selection == 'full-dev':
                selection_rows = dev_rows
            else:
                raise ValueError(f'Unknown checkpoint selection protocol: {checkpoint_selection}')
            dev_selection = summarize_prompt_scores(selection_rows)
            removed_count = len(controller.removed_heads(mask))
            kept_count = controller.total_heads - removed_count
            selection_score = paper_epoch_selection_score(float(dev_selection['gold_minus_distractor_logit']), kept_count, head_count_weight)
            epoch_record = {'epoch': epoch, 'loss': float(np.mean(losses)), 'effect_loss': float(np.mean(effects)), 'mean_keep_probability': float(np.mean(keeps)), 'kept_heads': kept_count, 'removed_heads': removed_count, 'paper_selection_score': selection_score, 'checkpoint_selection': checkpoint_selection, 'dev_selection': dev_selection, 'dev_full': dev_full}
            history.write(json.dumps(epoch_record, ensure_ascii=False) + '\n')
            history.flush()
            print(f"epoch={epoch} loss={epoch_record['loss']:.4f} selection_gap={float(dev_selection['gold_minus_distractor_logit']):.4f} full_dev_gap={float(dev_full['gold_minus_distractor_logit']):.4f} kept={kept_count} removed={removed_count} paper_score={selection_score:.4f}", flush=True)
            if selection_score > best_score:
                best_score = selection_score
                removed = controller.removed_heads(mask)
                best_payload = {'epoch': epoch, 'selection_rule': 'dev_gold_minus_distractor_logit + kept_heads * head_count_weight', 'checkpoint_selection': checkpoint_selection, 'selection_example_count': len(selection_rows), 'head_count_weight': head_count_weight, 'paper_selection_score': selection_score, 'shape': [controller.n_layers, controller.n_heads], 'kept_heads': controller.total_heads - len(removed), 'removed_head_count': len(removed), 'keep_mask': mask.cpu().tolist(), 'removed_heads': [{'layer': item.layer, 'head': item.head} for item in removed], 'dev_selection': dev_selection, 'dev_full': dev_full}
                write_json(output_dir / 'best_mask.json', best_payload)
    if best_payload is None:
        raise RuntimeError('Head search finished without producing a mask')
    best_mask = torch.tensor(best_payload['keep_mask'], device=controller.mask_logits.device, dtype=torch.float32)
    final_evaluation = evaluate_mask_conditions(model, tokenizer, controller, test_examples, batch_size=batch_size, mask=best_mask, output_dir=output_dir)
    report = {'best': best_payload, 'test': final_evaluation}
    write_json(output_dir / 'head_search_report.json', report)
    return report
