# H100 실행 순서 (runbook)

본 실행(논문 §5 8단계)까지 H100 한 장에서 할 일을 순서대로 적었다. 단계마다 "통과 조건"을
만족해야 다음 단계로 간다. 1–6단계는 구현 검증(§4.6)이다. 그 출력에서 효과 크기, AUC, pass rate,
탐지기 순위, 검정력을 계산하거나 보지 않는다. 연구 데이터는 8단계에서만 생긴다.

모든 명령은 저장소의 `pipeline/` 디렉터리에서 실행한다.

## 0. 코드 가져오기

- H100에 이 저장소를 받고, 실행할 커밋을 체크아웃한다. 추적 파일에 변경이 없어야 한다.
- **통과 조건**: `git status --short --untracked-files=no` 출력이 비어 있다.

## 1. 환경

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-h100.txt
pip install -e .
```

- Llama-3.1-8B-Instruct는 접근 승인이 필요한 모델이다. H100 계정의 Hugging Face 토큰으로 접근이
  되는지 2단계가 확인한다. 토큰 설정은 직접 한다.

## 2. 사전 점검 (1차)

```bash
python scripts/preflight_h100.py
```

- 이 시점에 FAIL이 나와도 되는 항목은 두 가지뿐이다.
  - AWQ 체크포인트 5개 (3단계에서 만든다)
  - Qwen2.5-32B, Olmo3.1-32B의 `sample_batch_size` (5단계에서 정한다)
- **통과 조건**: 그 밖의 행이 모두 PASS다. 특히 Python 3.12, 패키지 버전, GPU 80GB, 디스크 여유,
  다섯 모델 revision(Llama 접근 포함), LiveCodeBench 파일.

## 3. AWQ 체크포인트 만들기

```bash
python scripts/quantize_model.py Qwen2.5-7B-Instruct
python scripts/quantize_model.py Llama-3.1-8B-Instruct
python scripts/quantize_model.py Olmo3-7B-Instruct
python scripts/quantize_model.py Qwen2.5-32B-Instruct
python scripts/quantize_model.py Olmo3.1-32B-Instruct
```

- 결과는 `data/quantized/<model>-awq/`에 생기고, 각각 `quantization_manifest.json`과
  `calibration_overlap_report.json`이 함께 생긴다.
- 32B 모델의 AWQ 생성이 80GB에서 끝나는지는 아직 확인한 적이 없다.
- **통과 조건**: 다섯 개 모두 끝나고 두 JSON 파일이 있다. overlap 수가 0이 아니면 기록하고 판단한다
  (loader는 보고서가 없을 때만 막는다).

## 4. 스모크 테스트 (20개 셀)

모델 5개 × 정밀도 4개(`bf16`, `bnb_int8`, `bnb_nf4`, `gptq_awq_int4`) 모두 실행한다.

```bash
python scripts/run_smoke_test.py --model <MODEL> --quant <QUANT>
python scripts/run_lcb_smoke_test.py --model <MODEL> --quant <QUANT>
```

- 32B 모델은 아직 묶음 크기가 없으므로 `run_smoke_test.py`에 `--sample-batch-size 2`를 붙인다
  (이 테스트는 샘플을 2개만 뽑는다).
- 출력은 `data/raw/validation/` 아래에만 쌓인다.
- **통과 조건**: 40번 모두 exit 0. 메모리 상한은 카드 용량(80GB)으로 잘린다.

## 5. 샘플 묶음 크기 측정

32B 두 모델은 네 정밀도 모두 후보 크기를 잰다.

```bash
python scripts/measure_sample_batch.py --model Qwen2.5-32B-Instruct --quant <QUANT> --batch-sizes 50 25 10 5
python scripts/measure_sample_batch.py --model Olmo3.1-32B-Instruct --quant <QUANT> --batch-sizes 50 25 10 5
```

7B/8B 세 모델은 50이 들어가는지와 재현성만 확인한다.

```bash
python scripts/measure_sample_batch.py --model <7B/8B MODEL> --quant <QUANT> --batch-sizes 50 --n-items 1
```

- 스크립트는 가장 긴 LCB 문항으로 샘플 50개를 뽑는다. 크기마다 GPU 메모리 최고치, 문항당 시간,
  생성 길이, 512토큰 상한 도달률을 기록한다. 같은 문항을 두 번 생성해 토큰과 로그확률이 완전히
  같은지(`reproducible`)도 기록한다.
- 추천값은 "메모리 부족 없음 + 재현성 통과 + 최고 예약 메모리 ≤ 카드의 90%"를 만족하는 가장 큰
  크기다. 90%는 경험값이 아니라 여유분으로 정한 값이다.
- **모델의 묶음 크기 = 네 정밀도 추천값 중 가장 작은 값.** 같은 모델은 모든 정밀도에서 같은 크기를
  쓴다(논문 §4.4).
- **통과 조건**: 20개 셀 모두 `reproducible: true`. 하나라도 false면 본 실행 전에 원인을 찾는다.
  같은 입력에서 결과가 달라지면 중단 후 재개한 결과도 달라지기 때문이다.
- **시간 추정**: 기록된 문항당 시간 × 1,597문항 × 셀 수로 전체 GPU 시간을 계산한다. 가장 긴
  문항으로 잰 값이라 실제보다 크게 나온다. 감당할 수 없으면 본 실행 전에 계획을 다시 정한다.

## 6. 묶음 크기 고정과 사전 점검 (최종)

- `src/qcd/models/registry.py`에서 두 32B 모델의 `sample_batch_size`를 5단계 값으로 채운다.
  7B/8B가 50에서 통과하지 못했다면 그 값도 고친다.
- 테스트를 돌리고 커밋한다. 이 커밋이 본 실행 코드다.

```bash
python -m pytest -q
python scripts/preflight_h100.py
```

- **통과 조건**: 테스트 통과, 사전 점검 exit 0(FAIL 없음), 작업 트리 깨끗함.
- 이 시점 이후로 모델·문항·채점·분석 구성을 바꾸지 않는다.

## 7. 본 실행 (연구 데이터)

셀 하나씩 별도 프로세스로 돌린다. tmux 같은 세션 안에서 실행한다.

```bash
python scripts/run_main.py --cell <MODEL>:<QUANT>
```

- 20개 셀을 모두 끝낼 때까지 반복한다. 출력은 `data/raw/main/`이다.
- 끊기면 **같은 명령을 다시 실행**한다. 끝난 셀은 모델을 불러오지 않고 건너뛰고, 쓰다 만 셀은
  쓰지 않은 25문항 단위 묶음부터 이어 간다. 이미 쓴 파일은 바뀌지 않는다.
- 셀이 끝나면 `data/raw/main/cells/<cell>/complete.json`이 생긴다.
- 메모리 부족 같은 운영 실패로 설정을 바꿔야 하면, 결과를 보기 전에 기록하고 다시 고정한다(§4.6).
  이미 쓴 셀의 데이터와 섞지 않는다.
- **통과 조건**: 20개 셀 모두 `complete.json`이 있다.

## 8. 데이터 내려받기와 분석

로컬에서:

```bash
scripts/sync_from_h100.sh <ssh-alias> -- --dry-run
scripts/sync_from_h100.sh <ssh-alias>
python scripts/run_analysis.py --help
```

- 분석은 20개 셀이 모두 끝난 본 실행 데이터만 받는다. 검증용 출력이나 일부만 끝난 데이터는 거부한다.
