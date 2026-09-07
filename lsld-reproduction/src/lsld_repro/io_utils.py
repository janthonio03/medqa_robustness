from __future__ import annotations
import json
import os
import platform
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TypeVar
import numpy as np
T = TypeVar('T')

def batches(items: Sequence[T], batch_size: int) -> Iterator[Sequence[T]]:
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]

def prepare_output_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def _jsonable(item: object) -> object:
    if is_dataclass(item):
        return asdict(item)
    if hasattr(item, 'item'):
        return item.item()
    return item

def write_json(path: str | Path, payload: object) -> None:
    with Path(path).open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_jsonable)

def write_jsonl(path: str | Path, records: Iterable[object]) -> None:
    with Path(path).open('w', encoding='utf-8') as handle:
        for record in records:
            payload = record.to_dict() if hasattr(record, 'to_dict') else record
            handle.write(json.dumps(payload, ensure_ascii=False, default=_jsonable) + '\n')

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def environment_metadata() -> dict[str, object]:
    from . import __version__
    metadata: dict[str, object] = {'lsld_reproduction': __version__, 'python': platform.python_version(), 'platform': platform.platform(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID')}
    try:
        import torch
        import transformers
        metadata.update({'torch': torch.__version__, 'transformers': transformers.__version__, 'cuda_available': torch.cuda.is_available(), 'cuda_version': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    except ImportError:
        pass
    return metadata
