# 논문 우선 재현 실행 지시서 — v0.4.0

대상 경로:

```text
/nas2/data/janthonio03/robust_medqa/med_prm/lsld
```

대상 서버 설정은 `batch_ugrad`, `ariel-v8`, GPU 1장, CPU 8개, 메모리
32 GiB다. v0.3 결과는 corrected-full-dev 비교용으로 보관하되 공개 코드의
checkpoint 선택을 재현한 결과로 사용하지 않는다.

## 1. 기존 파일 위에 v0.4 설치

`env`, `data`, `hf_cache`, `outputs`, `logs`는 삭제하지 않는다. 프로젝트 루트에
새 ZIP을 올린 뒤 다음을 실행한다.

```bash
cd /nas2/data/janthonio03/robust_medqa/med_prm/lsld

unzip -o lsld-reproduction-v0.4.0.zip -d .

/nas2/data/janthonio03/robust_medqa/med_prm/lsld/env/bin/python \
  -m pip install -e '.[dev]'
```

## 2. 설치와 도움말 확인

`micromamba run` 대신 환경 안의 실행 파일을 직접 호출한다. 그러면 성공 출력이
터미널에 바로 보인다.

```bash
cd /nas2/data/janthonio03/robust_medqa/med_prm/lsld

/nas2/data/janthonio03/robust_medqa/med_prm/lsld/env/bin/python -c \
  "import lsld_repro; print(lsld_repro.__version__)"

/nas2/data/janthonio03/robust_medqa/med_prm/lsld/env/bin/pytest -q

COLUMNS=200 /nas2/data/janthonio03/robust_medqa/med_prm/lsld/env/bin/lsld \
  head-search --help | grep -E 'head-data-protocol|head-count-weight|checkpoint-selection'
```

정상 기준은 버전 `0.4.0`, test 통과, 세 CLI 옵션 표시다.

## 3. GPU 없이 데이터 프로토콜 확인

```bash
cd /nas2/data/janthonio03/robust_medqa/med_prm/lsld

/nas2/data/janthonio03/robust_medqa/med_prm/lsld/env/bin/python \
  scripts/verify_head_protocol.py
```

반드시 다음 수와 `"passed": true`가 나와야 한다.

```text
train: 360
dev:    40
test:  276
```

이 수가 다르면 500 epochs 작업을 제출하지 않는다.

## 4. 현재 작업 확인

```bash
squeue -u "$USER" \
  --format='%.12i %.24j %.10T %.12M %.12l %.15R'
```

현재 돌아가는 이전 버전 작업은 새 코드 검증에 사용할 수 없다. 작업 ID를 정확히
확인한 후 필요하면 `scancel 작업ID`로 종료한다. 다른 작업 ID는 취소하지 않는다.

## 5. full-data 1-epoch 호환성 검사

이 검사는 축소 데이터가 아니라 논문 재현과 동일한 `360/40/276`, batch 16을
사용하고 epoch만 1로 줄인다.

```bash
cd /nas2/data/janthonio03/robust_medqa/med_prm/lsld
mkdir -p logs

JOB_ID=$(sbatch --parsable slurm/03_head_search_timing.sbatch)
echo "JOB_ID=$JOB_ID"
squeue -j "$JOB_ID"
```

종료 후 확인한다.

```bash
cat "logs/${JOB_ID}.out"

cat outputs/head_search_protocol_check_v040/data_manifest.json

sacct -j "$JOB_ID" \
  --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS,NodeList%15
```

정상 기준:

- `State=COMPLETED`, `ExitCode=0:0`
- `data_manifest.json`에 `official-code`와 `360/40/276`
- 로그에 `epoch=1`
- `test_measurements.jsonl`이 276줄
- CUDA OOM 없음

줄 수 확인:

```bash
wc -l \
  outputs/head_search_protocol_check_v040/train_examples.jsonl \
  outputs/head_search_protocol_check_v040/dev_examples.jsonl \
  outputs/head_search_protocol_check_v040/test_examples.jsonl \
  outputs/head_search_protocol_check_v040/test_measurements.jsonl
```

## 6. seed-0 500-epoch 본 실험

1-epoch 검사가 통과한 뒤에만 실행한다.

```bash
cd /nas2/data/janthonio03/robust_medqa/med_prm/lsld
mkdir -p logs

JOB_ID=$(sbatch --parsable slurm/03_head_search.sbatch)
echo "JOB_ID=$JOB_ID"
squeue -j "$JOB_ID"
```

설정은 다음과 같다.

```text
model                 meta-llama/Llama-3.1-8B
head data             official-code, 360/40/276
epochs                500
batch size            16
optimizer             AdamW
learning rate         1.0
sparsity lambda       1.0
Gumbel temperature    1.0
selection source      released-code final dev batch (8/40)
selection             selection gap + kept heads × 0.1
seed                   0
train shuffle         false
```

공개 코드의 `evaluate()`는 batch loop가 끝난 뒤 logit difference를 한 번만
추가하므로 checkpoint 선택에는 마지막 dev batch만 반영된다. Dev 40개와 batch
16에서는 마지막 8개다. v0.4는 논문 결과 호환을 위해 이 8개로 선택 점수를
계산하지만, `history.jsonl`의 `dev_full`에는 40개 전체 평균도 함께 저장한다.
최종 test 평가는 276개 전체를 사용한다.

진행 확인:

```bash
tail -n 5 outputs/head_search_capital_v040_seed0/history.jsonl

squeue -j "$JOB_ID"
```

## 7. 종료 후 논문 Table 2 비교

```bash
cat outputs/head_search_capital_v040_seed0/data_manifest.json

cat outputs/head_search_capital_v040_seed0/head_search_report.json

/nas2/data/janthonio03/robust_medqa/med_prm/lsld/env/bin/python \
  scripts/compare_table2.py

wc -l outputs/head_search_capital_v040_seed0/test_measurements.jsonl

sacct -j "$JOB_ID" \
  --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS,NodeList%15
```

논문의 country–capital 비교 목표는 다음과 같다.

| 항목 | 논문 Table 2 |
|---|---:|
| 제거 head | 36 |
| all-heads with-context logit gap | 약 7.69 |
| masked with-context logit gap | 약 13.20 |
| distractor rank | 약 37.5 → 1289.6 |

확률적 학습이므로 정확히 36개와 일치하는 것만으로 성공 여부를 판정하지 않는다.
동일 방향의 gap 증가, distractor rank 하락(숫자는 증가), 유의한 paired test가 함께
나오는지 확인한다.

Appendix D의 5회 반복 비교 목표는 제거 head 60–96개와 pairwise Jaccard
0.317–0.548이다. 이는 Table 2의 대표 run 36개와 별개의 반복실험 결과다.

## 8. 5회 안정성 실험

논문 부록의 반복 안정성을 확인할 때만 실행한다. seed-0 결과를 먼저 검토한 뒤,
아래 array로 seed 1–4를 추가해 총 5회를 구성한다.

```bash
cd /nas2/data/janthonio03/robust_medqa/med_prm/lsld
mkdir -p logs

ARRAY_JOB_ID=$(sbatch --parsable slurm/04_head_search_five_seeds.sbatch)
echo "ARRAY_JOB_ID=$ARRAY_JOB_ID"
squeue -j "$ARRAY_JOB_ID"
```

모든 run이 끝난 뒤 Jaccard overlap을 계산한다.

```bash
/nas2/data/janthonio03/robust_medqa/med_prm/lsld/env/bin/python \
  scripts/summarize_head_runs.py \
  outputs/head_search_capital_v040_seed*/best_mask.json
```

## 9. MedQA로 넘어가는 조건

다음을 확보한 뒤에만 객관식 의료 데이터용 scoring을 설계한다.

- seed-0 500-epoch 작업의 정상 종료
- `data_manifest.json`의 `360/40/276`
- `run_config.json`의 checkpoint selection이 `released-code-last-batch`
- `best_mask.json`의 selection example count가 `8`
- development-selected `best_mask.json`
- 276개 `test_measurements.jsonl`
- `head_search_report.json`의 paired test
- 논문 Table 2와 방향·규모 비교
- 실행 코드 ZIP, Slurm 로그, `run_config.json`
