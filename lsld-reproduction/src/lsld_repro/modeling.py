from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .io_utils import batches

@dataclass(frozen=True)
class TokenTarget:
    text: str
    token_id: int
    token_text: str
    all_token_ids: tuple[int, ...]

def resolve_dtype(name: str) -> torch.dtype:
    if name == 'float32':
        return torch.float32
    if name == 'float16':
        return torch.float16
    if name == 'bfloat16':
        return torch.bfloat16
    if name != 'auto':
        raise ValueError(f'Unknown dtype: {name}')
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32

def load_model_and_tokenizer(model_name: str, *, dtype: str='auto', device: str='cuda', attn_implementation: str='eager'):
    if device.startswith('cuda') and (not torch.cuda.is_available()):
        raise RuntimeError('CUDA was requested but torch.cuda.is_available() is false')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=resolve_dtype(dtype), attn_implementation=attn_implementation, low_cpu_mem_usage=True)
    model.to(device)
    model.eval()
    return (model, tokenizer)

def first_leading_space_token(tokenizer, text: str) -> TokenTarget:
    ids = tokenizer.encode(' ' + text.strip(), add_special_tokens=False)
    if not ids:
        raise ValueError(f'Target produced no tokens: {text!r}')
    token_id = int(ids[0])
    return TokenTarget(text=text, token_id=token_id, token_text=tokenizer.decode([token_id]), all_token_ids=tuple((int(x) for x in ids)))

def last_token_logits(model, tokenizer, prompts: Sequence[str]) -> torch.Tensor:
    encoded = tokenizer(list(prompts), return_tensors='pt', padding=True, truncation=False)
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    lengths = encoded['attention_mask'].sum(dim=1) - 1
    row_index = torch.arange(len(prompts), device=device)
    base_model = getattr(model, 'model', None)
    if base_model is None:
        base_model = getattr(model, 'transformer', None)
    lm_head = getattr(model, 'lm_head', None)
    if base_model is not None and lm_head is not None:
        hidden = base_model(**encoded, use_cache=False, return_dict=True).last_hidden_state
        return lm_head(hidden[row_index, lengths])
    output = model(**encoded, use_cache=False, return_dict=True).logits
    return output[row_index, lengths]

def target_rank(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    selected = logits.gather(1, token_ids[:, None])
    return (logits > selected).sum(dim=1) + 1

def gather_target_scores(logits: torch.Tensor, token_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    log_probs = logits.log_softmax(dim=-1)
    selected_logits = logits.gather(1, token_ids[:, None]).squeeze(1)
    selected_log_probs = log_probs.gather(1, token_ids[:, None]).squeeze(1)
    return {'logit': selected_logits, 'log_probability': selected_log_probs, 'probability': selected_log_probs.exp(), 'rank': target_rank(logits, token_ids)}

@torch.no_grad()
def score_prompt_targets(model, tokenizer, prompts: Sequence[str], target_ids: Sequence[int], *, batch_size: int) -> list[dict[str, float | int]]:
    output: list[dict[str, float | int]] = []
    paired = list(zip(prompts, target_ids, strict=True))
    for batch in batches(paired, batch_size):
        batch_prompts = [item[0] for item in batch]
        batch_ids = torch.tensor([item[1] for item in batch], device=next(model.parameters()).device)
        logits = last_token_logits(model, tokenizer, batch_prompts)
        scores = gather_target_scores(logits, batch_ids)
        for logit, log_prob, rank in zip(scores['logit'], scores['log_probability'], scores['rank'], strict=True):
            output.append({'logit': float(logit.float().cpu()), 'log_probability': float(log_prob.float().cpu()), 'probability': float(log_prob.float().exp().cpu()), 'rank': int(rank.cpu())})
    return output
