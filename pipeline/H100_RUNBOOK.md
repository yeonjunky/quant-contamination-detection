# H100 실행 순서 (runbook)

본 실행(논문 §5 8단계)까지 H100 한 장에서 할 일을 순서대로 적었다. 단계마다 "통과 조건"을
만족해야 다음 단계로 간다. 1–6단계는 구현 검증(§4.6)이다. 그 출력에서 효과 크기, AUC, pass rate,
탐지기 순위, 검정력을 계산하거나 보지 않는다. 연구 데이터는 8단계에서만 생긴다.

모든 명령은 저장소의 `pipeline/` 디렉터리에서 실행한다.

## 무엇을 동시에 돌리고, 무엇을 하나씩 돌리나

- **동시에 돌린다**
  - 모델·데이터 내려받기(1단계 끝)는 GPU를 쓰지 않으므로 다섯 모델을 한꺼번에 받고, 그동안 2단계
    점검을 한다.
  - 본 실행 중 코드 채점(CPU)은 다음 문항의 GPU 생성과 겹쳐서 돈다(7단계).
  - 로컬 쪽: 끝난 셀의 데이터 동기화(8단계 명령)는 본 실행이 도는 동안에도 할 수 있다. 분석은 20개
    셀이 다 끝나야 돈다.
- **하나씩 돌린다 (GPU를 쓰는 단계)**
  - AWQ 생성, 스모크 테스트, 묶음 크기 측정, 본 실행 셀은 한 번에 하나만 GPU에 올린다.
  - 이유: 묶음 크기와 메모리 상한은 GPU 80GB를 혼자 쓴다는 전제로 잰다. 두 작업이 GPU를 나눠 쓰면
    측정값이 틀리고, 본 실행에서 메모리 부족이 날 수 있다. 7B 셀 두 개를 같이 올리는 것도 같은
    이유로 하지 않는다.

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

- torch는 `requirements-h100.txt`의 `torch==2.13.0`이 PyPI에서 설치된다. 별도 wheel 주소
  (`--index-url`)는 쓰지 않는다. 2026-08-15 H100 기록(`envs/local-smoke-freeze.txt`)과 같은 버전이며,
  그 기록에는 CUDA 13.0 패키지가 함께 있다. torch가 스스로 붙이는 `+cu130` 같은 꼬리표는 2단계
  점검이 무시하고, torch가 어떤 CUDA 버전으로 빌드됐는지는 `cuda` 행에 따로 보여 준다.
- 환경이 준비되면 내려받기를 백그라운드로 시작한다. registry에 고정된 revision의 다섯 모델과
  LiveCodeBench, AWQ 보정 데이터를 동시에 받는다. 다섯 모델 bf16은 약 173GB다(추정).

```bash
nohup python scripts/prefetch_models.py > prefetch.log 2>&1 &
```

- 내려받는 동안 2단계 점검을 한다. 3단계는 `prefetch.log`의 모든 줄이 `done`이고 프로세스가 exit 0으로
  끝난 뒤 시작한다. `FAIL` 줄이 있으면(예: Llama 접근 권한) 고친 뒤 다시 실행한다. 이미 받은 파일은 다시
  받지 않는다.
- Llama-3.1-8B-Instruct는 접근 승인이 필요한 모델이다. H100 계정의 Hugging Face 토큰으로 접근이
  되는지 2단계가 확인한다. 토큰 설정은 직접 한다.
- 채점에 영향을 주는 환경 변수 `EVALPLUS_MAX_MEMORY_BYTES`, `EVALPLUS_TIMEOUT_PER_TASK`,
  `HUMANEVAL_OVERRIDE_PATH`, `MBPP_OVERRIDE_PATH`는 **설정하지 않는다.** 값이 실행 기록에 들어가므로
  셀마다 다르면 그 셀은 거부된다. 특히 `EVALPLUS_TIMEOUT_PER_TASK`는 설정하면 evalplus 0.3.1이 문자열과
  숫자를 비교하다 오류를 낼 것으로 보인다(코드를 읽어 판단했고, 실행해 보지는 않았다).

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
- 출력은 `data/raw/validation/` 아래에만 쌓이고, 셀마다 따로 폴더가 생긴다
  (`smoke_test/<MODEL>/<QUANT>/`, `lcb_smoke_test/<MODEL>-<QUANT>/`). `pip freeze` 기록도 그 실행의 폴더
  (`data/raw/validation/smoke_test/<MODEL>/<QUANT>/pip-freeze.txt`)에 쓰이고, 그 경로가 manifest에 남는다.
  추적 파일 `envs/local-smoke-freeze.txt`는 건드리지 않으므로 스모크 테스트 뒤에도 작업 트리가
  깨끗하다.
- `run_smoke_test.py`는 5개 문항의 greedy 출력이 모두 512토큰 상한까지 갔으면
  `greedy_outputs_not_all_at_512_cap` 항목을 실패로 표시한다. 가장 짧은 HumanEval 문항에서 한 번도
  멈추지 않았다면 모델의 종료 토큰 설정이 틀렸을 가능성이 크다. 점수가 아니라 실행의 성질을 보는
  점검이다.
- 스모크 테스트는 한 프로세스 안에서 차례로 채점한다. 본 실행의 채점 방식(작업 프로세스 4개)은 따로
  점검한다. HumanEval+/MBPP+ 문항 40개의 정답 코드를 그 작업 프로세스들과 한 프로세스 안에서 각각
  채점해, 모두 만점인지 본다. 모델 출력은 쓰지 않고 데이터도 남기지 않는다.

```bash
python scripts/check_sandbox_pool.py
```

- **통과 조건**: 40번 모두 exit 0(종료 토큰 점검 포함), `check_sandbox_pool.py`가 `PASS`(exit 0).
  메모리 상한은 카드 용량(80GB)으로 잘린다. `git status --short --untracked-files=no` 출력이 여전히
  비어 있다.

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

- 스크립트는 가장 긴 LCB 문항으로 본 실행이 문항마다 하는 일을 그대로 한다. 샘플 50개를 뽑고,
  이어서 문항 프롬프트의 로그확률을 계산하는 순전파를 한 번 한다(그 값은 읽지 않고 버린다).
  크기마다 두 번 돈다.
  - `normal_decoding`: 본 실행과 같은 디코딩. 같은 문항을 두 번 생성해 토큰과 로그확률이 완전히
    같은지(`reproducible`)를 기록하고, 모든 생성의 토큰과 로그확률로 만든 SHA-256 값
    (`generations_sha256`)을 남긴다. 점수가 아니라 생성 결과의 지문이다.
  - `forced_length`: 종료 토큰을 막아 모든 줄이 512토큰까지 생성되게 한다. 이 메모리 최고치가 그
    길이의 문항에서 나올 수 있는 최악의 값이다. 종료 토큰을 막는 장치는 이 스크립트에서만 쓰고 본
    실행에서는 쓰지 않는다.
- 기록의 숫자마다 어느 쪽에서 나왔는지 적혀 있다.
- 추천값은 "메모리 부족 없음 + 재현성 통과 + `forced_length`의 모든 줄이 512토큰 도달 +
  `forced_length` 최고 예약 메모리 ≤ 카드의 90%"를 만족하는 가장 큰 크기다. 90%는 경험값이 아니라
  여유분으로 정한 값이다.
- 어떤 크기도 이 조건을 못 맞추면 스크립트가 exit 1로 끝나며 더 작은 크기로 다시 재라고 알려 준다.
  그때는 `--batch-sizes 4 3 2 1`로 다시 잰다.
- **모델의 묶음 크기 = 네 정밀도 추천값 중 가장 작은 값.** 같은 모델은 모든 정밀도에서 같은 크기를
  쓴다(논문 §4.4).
- **다른 프로세스에서 다시 재기.** 재개는 새 프로세스에서 일어나므로, 32B 셀 하나와 7B 셀 하나를
  골라 처음과 같은 인자로 새 프로세스에서 한 번 더 잰다. 결과는 다른 폴더에 쓰고 첫 기록과 비교한다.

```bash
python scripts/measure_sample_batch.py --model <MODEL> --quant <QUANT> <처음과 같은 --batch-sizes, --n-items> \
    --output-dir ../data/raw/validation/sample_batch_measurement/<MODEL>-<QUANT>-repeat \
    --compare-to ../data/raw/validation/sample_batch_measurement/<MODEL>-<QUANT>/batch_measurement.json
```

- **통과 조건**: 20개 셀 모두 `reproducible: true`, 그리고 다시 잰 두 셀의 비교가 "all digests match"로
  끝난다(exit 0). 하나라도 어긋나면 본 실행 전에 원인을 찾는다. 같은 입력에서 결과가 달라지면 중단
  후 재개한 결과도 달라지기 때문이다.
- **시간 추정 (생성만)**: `normal_decoding`에 기록된 문항당 시간 × 1,597문항 × 셀 수로 생성 시간을
  계산한다. 가장 긴 문항으로 잰 값이라 실제보다 크게 나온다. 이 값에는 코드 채점(샌드박스) 시간이
  빠져 있다. 채점은 GPU 생성이 끝난 뒤 문항마다 테스트를 하나씩 돌리므로, 시간 전체는 7단계 첫 셀의
  기록으로 다시 계산한다. 생성 시간만으로도 감당할 수 없으면 본 실행 전에 계획을 다시 정한다.

## 6. 묶음 크기 고정과 사전 점검 (최종)

- `src/qcd/models/registry.py`에서 두 32B 모델의 `sample_batch_size`를 5단계 값으로 채운다.
- 7B/8B 세 모델의 50은 논문(§4.4)이 정한 값이다. 5단계에서 7B/8B 모델이 50을 통과하지 못했으면
  여기서 멈춘다. 그 값을 바꾸는 것은 registry 수정만으로 끝나지 않는 실험 절차 변경이고, 논문을
  먼저 고쳐야 한다.
- 테스트를 돌리고 커밋한다. 이 커밋이 본 실행 코드다.

```bash
python -m pytest -q
python scripts/preflight_h100.py --min-free-gb 50
```

- 3–4단계에서 모델을 이미 내려받았으므로, 최종 점검의 디스크 기준은 본 실행 출력이 들어갈 여유만
  본다. 50GB는 출력 크기 추정(약 17GB, 평균 생성 길이 300토큰 가정)에 여유를 더한 값이다.
- **통과 조건**: 테스트 통과, 사전 점검 exit 0(FAIL 없음), 작업 트리 깨끗함.
- 이 시점 이후로 모델·문항·채점·분석 구성을 바꾸지 않는다.

## 7. 본 실행 (연구 데이터)

셀 하나씩 별도 프로세스로 돌린다. tmux 같은 세션 안에서 실행한다.

```bash
python scripts/run_main.py --cell <MODEL>:<QUANT>
```

- 20개 셀을 모두 끝낼 때까지 반복한다. 출력은 `data/raw/main/`이다.
- 코드 채점은 별도 작업 프로세스 4개가 맡고, 그동안 주 프로세스는 다음 문항을 GPU로 생성한다. 그래서
  프로세스 목록에는 셀마다 부모 1개와 자식 4개가 보인다. 행의 내용과 순서는 한 문항씩 차례로 채점할
  때와 같다(Mac에서 가짜 채점기로 확인). 실제 채점에서는 동시에 도는 채점 4개가 CPU를 나눠 쓰므로,
  시간 제한에 걸리기 직전인 테스트는 통과·실패가 바뀔 수 있다. HumanEval+/MBPP+ 정답 코드 60개는 Mac에서
  차례 채점과 결과가 같았다. H100에서는 작업 프로세스가 evalplus의 테스트 프로세스를 `spawn`으로 띄워 정답
  코드의 base 테스트가 전부 시간 초과됐고, 작업 프로세스 안에서는 `fork`로 띄우도록 고쳤다(2026-10-10).
  LCB에서는 확인하지 않았다. 그래서 각 셀의 시작 기록에 CPU 수와
  부하를 남긴다. 작업 프로세스 수 4는 실행 기록에 들어가므로 바꾸면 기존 본 실행
  폴더에서 거부된다. 작업 프로세스에서 오류가 나면 그 셀이 멈추고, 0점으로 기록되지 않는다.
- 첫 셀의 첫 묶음(25문항)이 쓰일 때까지 지켜보고, `generations` 파일이 생기는지와 오류가 없는지
  확인한다.
- 끊기면 **같은 명령을 다시 실행**한다. 끝난 셀은 모델을 불러오지 않고 건너뛰고, 쓰다 만 셀은
  쓰지 않은 25문항 단위 묶음부터 이어 간다. 이미 쓴 문항 묶음 파일은 건드리지 않는다. 다만 공용
  파일인 `items.parquet`, `model_item_labels.parquet`와 그 셀의 채팅 템플릿 파일
  (`chat_templates.<셀>.parquet`)은 다시 쓰이며, 내용은 이전과 같다.
- 셀이 처음 시작하면 `data/raw/main/cells/<cell>/started.json`, 이어서 돌릴 때마다 `resumed-<시각>.json`,
  끝나면 `complete.json`이 생긴다. 모두
  커밋, 패키지 버전, 호스트, GPU 이름을 기록한다. 시작 기록에는 CPU 수와 부하도 들어간다. 셀마다 시작할 때 약 1분이 더 걸린다(LCB 문항
  읽기와 숨은 테스트 수 세기).
- **본 실행은 첫 셀을 시작한 커밋과 패키지 버전에 묶인다.** 다른 커밋, 커밋 안 된 변경, 다른 패키지
  버전, 다른 채점 환경 변수로 셀을 시작하면 거부된다. `git pull`이나 `pip install`을 하지 않는다.
- **시간 다시 계산.** 채점이 생성과 겹쳐 돌기 때문에 시간 열을 더하면 실제보다 크게 나온다. 실제
  경과 시간으로 잰다. 첫 셀의 두 번째 묶음 파일(`raw/generations.<셀>-00001.parquet`)이 생긴 시각에서
  `started.json`의 `started_at_utc`를 빼면 50문항 걸린 시간이다. 이 값 × (1,597 ÷ 50)이 그 셀 하나의
  시간이다. 7B/8B와 32B는 속도가 다르므로 첫 32B 셀에서 한 번 더 잰다. 파일 시각과 시작 기록만 보고,
  점수나 통과 여부 열은 보지 않는다. 참고로 Mac에서 가짜 모델로 잰
  HumanEval+/MBPP+ 채점은 문항당 3–4초였다(정답 코드를 매번 다시 실행). H100 값은 재지 않았다.
- **설정을 바꿔야 할 때** (메모리 부족 같은 운영 실패): 결과를 보기 전에 원인과 바꿀 내용을 기록한다.
  그다음 `data/raw/main`을 `data/raw/main_aborted_<날짜>`로 옮기고, 코드를 고쳐 커밋하고, 6단계부터
  다시 한다. 새 본 실행은 20개 셀을 처음부터 돈다. 옮긴 폴더의 데이터는 분석에 쓰지 않는다(§4.6).
- **통과 조건**: 20개 셀 모두 `complete.json`이 있다.

## 8. 데이터 내려받기와 분석

로컬에서:

```bash
scripts/sync_from_h100.sh <ssh-alias> -- --dry-run
scripts/sync_from_h100.sh <ssh-alias>
python scripts/run_analysis.py ../data/raw/main
```

- 첫 줄은 미리 보기다(rsync `--dry-run`). `--` 다음 인자는 rsync에 그대로 넘어가고, H100 쪽 저장소
  경로는 기본값 `~/quant-contamination-detection`을 쓴다. 저장소가 다른 곳에 있으면
  `scripts/sync_from_h100.sh <ssh-alias> <원격 저장소 경로> -- --dry-run`처럼 별칭 바로 뒤에 경로를 쓴다.
- 기본은 `data/raw/main/`만 받는다. 검증용 출력은 `--with-validation`(둘 다) 또는
  `--validation-only`를 별칭 앞에 붙여서 따로 받는다.
- 분석 결과는 `../data/raw/main/analysis/`에 쓰인다.
- 분석은 20개 셀이 모두 끝난 본 실행 데이터만 받는다. 검증용 출력, 일부만 끝난 데이터, 커밋이
  다른 완료 기록은 거부한다.
