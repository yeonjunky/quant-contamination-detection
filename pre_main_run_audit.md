# 본 실행 전 준비도 감사 (읽기 전용)

- **감사 일자**: 2026-09-23
- **대상 HEAD**: `af960eb9c58a709c1d983aa6fc426bb95d741f07`
- **작업 트리**: dirty — `pipeline/src/qcd/detectors/cdd.py`, `pipeline/tests/test_detectors.py`
- **범위**: 읽기 전용 감사. 파일 수정·커밋·푸시·H100 접속·모델 다운로드·AWQ 생성·대규모 코퍼스 스캔 없음. `run_main.py` 미실행
- **성격**: 이 문서는 구현 준비도 판정이며 연구 결과가 아니다. 드라이런·스모크 출력에서 효과 크기, AUC, pass rate, 상관, 검정력, 탐지기 순위를 계산하거나 해석하지 않았다

---

## 1. 최종 판정

| 항목 | 판정 |
|---|---|
| 연구 설계 | **Ready** — 이번 감사에서 C1–C4, 모델, 문항 집합, 탐지기 우선순위를 바꿀 근거는 나오지 않았다 |
| 로컬 분석 코드 | **Needs revision** — 분석 계층(confirmatory / conditional_logit / coverage / run_analysis)은 Ready. 탐지기 계층은 HEAD 커밋 자체가 자기모순이라 커밋이 필요하다 |
| H100 main run | **NO-GO** — 차단 사유 3건 (P0-1, P0-2, P0-3) |
| Olmo/TRACER | **Parallel incomplete** — GPU 본 채점을 막지는 않지만 논문 완성을 막는다 |

핵심 결론: **논문이 기술한 CDD 정의를 구현한 코드가 커밋되어 있지 않고, manifest는 그 차이를 기록할 방법이 없으며, 현재 생성 경로는 배치 없이 1건씩 도는 163만 회 생성이라 단일 H100에서 계획대로 완주할 수 없다.**

---

## 2. 발견 사항

| # | 우선순위 | 위치 | 현재 상태 | 위험 | 근거 | 권장 조치 | 본 실행 차단 |
|---|---|---|---|---|---|---|---|
| 1 | **P0** | `pipeline/src/qcd/detectors/cdd.py:69` | HEAD 커밋의 코드는 `alpha * max_tokens`. 수정은 커밋되지 않은 작업 트리에만 존재 | HEAD를 체크아웃해 실행하면 논문과 다른 CDD 점수가 나옴 | HEAD의 `constants.py:67`은 이미 `"actual-max-truncated-length-v2"`, `paper/paper_draft.md:683-686`·`paper_draft_ko.md:646-650`·`pipeline/README.md:13`·`revision_provenance.md:477`이 모두 수정본 정의를 기술. 코드만 구버전 | 작업 트리 변경을 커밋 | **예** |
| 2 | **P0** | `pipeline/src/qcd/io/manifest.py:80-87`, `scripts/run_main.py` | `git rev-parse HEAD`만 기록. dirty flag·diff digest 없음. `run_main.py`에 clean 게이트 없음 | 지금 상태로 본 실행하면 manifest에 `af960eb`이 적히지만 실제 실행 코드는 그 커밋이 아님 → 재현 불가 | `get_git_commit_hash()`가 `rev-parse HEAD` 단일 호출. `run_main.py` 전체 64줄에 게이트 없음 | 4절 최소/강화 수정안 | **예** |
| 3 | **P0** | `pipeline/src/qcd/models/loader.py:495-546`, `generation/sampler.py:49-56` | 생성이 배치 없이 1건씩(`outputs.sequences[0]`), 샘플 50개를 루프로 순차 생성 | 총 1,628,940회 생성. README가 기록한 7B/nf4 실측(3생성 포함 문항당 10–19초)으로 환산하면 단일 H100에서 수천 GPU-시간 | 5절 검산 | **예** |
| 4 | **P1** | `pipeline/README.md:148-150` | Llama-3.1-8B, Qwen2.5-32B, Olmo3.1-32B 세 arm이 실하드웨어에서 한 번도 실행된 적 없음 (저장소 자신의 기록) | 본 실행 중 로딩·메모리·추출 실패가 처음 드러남 | README "Still not empirically exercised" 절 | 다섯 arm × 네 정밀도 전부 스모크 | 예 |
| 5 | **P1** | `pipeline/scripts/run_smoke_test.py:78-99` | AWQ 32B의 허용 메모리 밴드 상한이 134 GB. 80 GB H100보다 큼 | 밴드 검사가 OOM을 사전에 못 잡음. 7B AWQ의 실측 peak이 bf16 폭(15–16 GB)이었으므로 32B AWQ는 ~65 GB로 추정 | `_UPPER_FACTOR=2.0`, AWQ ceiling_basis = bf16 폭. 32.5B × 2 × 2 + 4 = 134 GB | 32B AWQ/bf16 실측 없이 GO 판정 금지 | 예 |
| 6 | **P1** | `data/quantized/` | 이 체크아웃에 존재하지 않음. 다섯 AWQ 체크포인트, `quantization_manifest.json`, `calibration_overlap_report.json` 전부 로컬 미확인 | AWQ rung이 로드 시점에 `FileNotFoundError`로 실패 | `loader.py:360-391`이 세 파일을 모두 강제. `ls data/quantized` → 없음 | H100 상태 확인 필요 (미검증) | 예 |
| 7 | **P1** | `pipeline/tests/` | 중단→resume 경로와 `config_hash` 거부 경로에 테스트 없음 | 본 실행 중단 후 재개가 조용히 다르게 동작해도 잡히지 않음 | `grep "already contains a different run configuration" tests/` → 0건 | resume 테스트 추가 | 아니오 (권장) |
| 8 | **P1** | `pipeline/src/qcd/detectors/cdd.py:68` | `l`이 문항·정밀도마다 달라짐 (실제 출력 길이에 의존). 현재 어디에도 `l`을 기록·보고하지 않음 | C1–C3의 CDD 대응 이동에 "길이 변화 성분"과 "peakedness 성분"이 섞임. 양자화가 출력 길이를 바꾸면 임계값 자체가 바뀜 | 구버전 `alpha*l_max`는 정밀도 간 임계값이 상수였음. 신버전은 아님 | `token_ids`가 전부 저장되므로 사후 재계산 가능 — 데이터 손실은 없음. 분석 측에서 조건별 `l` 분포를 보고하도록 추가 | 아니오 |
| 9 | **P1** | `pipeline/src/qcd/detectors/cdd.py` | greedy와 50샘플이 전부 빈 출력이면 `l=0`, 임계값 0, `Peak=1.0` | 빈 출력률이 정밀도마다 다르면 C1–C3을 교란 (nf4에서 즉시 EOS가 나올 수 있음) | 직접 실행 확인: `peakedness([], [[],[],[]])` → `1.0` | 구버전에서도 동일한 동작이므로 이번 수정이 만든 문제는 아님. 빈 출력률을 정밀도별로 기록 | 아니오 |
| 10 | **P1** | `pipeline/OLMO_GROUND_TRUTH.md` | Olmo3.1-32B 사전학습 mix(`allenai/dolma3_mix-6T`)에 스캔 revision이 고정돼 있지 않음 | 32B 코퍼스 결과를 과학적 결과로 쓸 수 없음 | 표에 "not pinned". `olmo_corpora.require_revision`이 거부 | Hub에서 revision 확정 후 기록 | 아니오 (논문 완성은 차단) |
| 11 | **P2** | `pipeline_implementation_log.md:107-109` | 아직 `α·l_max`로 기술 | 문서 간 불일치 | grep 확인 | 문서 동기화 시 함께 수정 | 아니오 |
| 12 | **P2** | `figures/fig_*.png` | 3장 모두 현재 수치와 불일치 | 없음 (두 드래프트 어디서도 참조하지 않음) | HEAD 커밋 `af960eb`이 `figures/README.md`에 파일별로 기록 | 재생성 여부는 별도 결정 | 아니오 |
| 13 | **P2** | `pipeline/scripts/run_main.py:42-43` | `--n-cdd-samples`, `--lcb-cutoff`가 게이트 없는 CLI 오버라이드 | n=50이 아닌 값으로 본 실행이 시작될 수 있음 | manifest에는 기록되고 `config_hash`에도 들어감 | §4.4의 "문서화·동결" 요구를 코드 게이트로 올릴지 결정 | 아니오 |
| 14 | **P2** | `pipeline/OLMO_GROUND_TRUTH.md`, `TRACER_REIMPLEMENTATION.md` | n-gram `n=13`, coverage `0.8`, `--candidates-per-item 5`, TRACER LLM 3단계의 모델 identity·디코딩 미동결 | 검색 설정이 결과를 본 뒤 바뀔 여지 | 두 문서가 스스로 "provisional / remain to be frozen"이라 기록 | Q1 결과를 보기 전에 동결 | 아니오 |

---

## 3. 본 실행 전 체크리스트

### 3.1 저장소·코드

| 항목 | 판정 | 근거 |
|---|---|---|
| HEAD가 `af960eb9c58a709c1d983aa6fc426bb95d741f07` | **PASS** | `git rev-parse HEAD` |
| 미커밋 변경 2개 파일 | **PASS** | `git status --short` → `M pipeline/src/qcd/detectors/cdd.py`, `M pipeline/tests/test_detectors.py` |
| 공백 오류 없음 | **PASS** | `git diff --check` rc=0 |
| 기존 main-study / validation 산출물 없음 | **PASS** | 저장소 어디에도 `data/` 없음. `find . -name "*.parquet"` → `.venv` 내부 테스트 픽스처뿐 |
| 유일한 분석 산출물 | **PASS** | `pipeline/analysis_artifacts/interval_coverage_check.json` 1개 |
| 코드와 문서가 같은 CDD 정의를 쓰는지 | **FAIL** | HEAD 코드만 구버전. 문서 4종·상수 1종은 신버전 |
| clean-commit 실행 게이트 | **FAIL** | `run_main.py`에 없음 |

### 3.2 CDD 정의 (원문 대조)

`pdfs/2603.03203.pdf`를 좌/우 컬럼 분리 추출해 §3.1을 직접 읽었다. 기억에 의존하지 않았다.

| 확인 항목 | 판정 | 원문 자구 / 확인 방법 |
|---|---|---|
| `l`이 고정 100이 아니라 실제 최대 길이 | **PASS** | 식 (1) 아래: `l = max{\|s\| : s ∈ S ∪ {s_t=0}} is the maximum sequence length across all samples` |
| 절단은 상한 (`l_max=100`) | **PASS** | `All sequences are tokenized using the model's BPE tokenizer and truncated to l_max=100 tokens` |
| greedy 1 + 샘플 50, star topology | **PASS** | `We use n=50, matching the original paper` / `CDD applies this in a star topology: the greedy reference s_t=0 is compared against each temperature sample s_i, rather than computing all pairwise distances` |
| 온도 0 / 0.8 | **PASS** | 원문 동일. `constants.py:55-57` 일치 |
| token-level Levenshtein | **PASS** | `the token-level Levenshtein edit distance (Levenshtein, 1965)` / `cdd.py:29-50`이 O(n·m) DP로 구현 |
| 비교가 `≤` | **PASS** | 식 (1) `I(ED(s_i, s_t=0) ≤ α·l)` |
| 작업 트리 코드가 위 정의와 일치 | **PASS** | `cdd.py:67-69` — 절단 후 greedy와 전체 샘플의 최대 길이 |
| 정확히 100토큰 경계 테스트 | **PASS** | `test_peakedness_threshold_boundary` (ED=5 통과, ED=6 실패). 직접 실행 확인 |
| 짧은 출력 경계 테스트 | **PASS** | `test_peakedness_short_outputs_use_actual_maximum_length` (l=10, 1 edit → 0.0) |
| greedy가 최대값 계산에 참여하는지 | **PASS** | `test_peakedness_uses_one_maximum_across_reference_and_all_samples` |
| 임계값이 정수가 아닐 때의 비교 규칙 | **PASS (문서화는 없음)** | 부동소수점 직접 검산: `0.05*100=5.0`, `0.05*20=1.0`, `0.05*40=2.0`, `0.05*60=3.0`, `0.05*80=4.0` 전부 정확. `4.999…` 위험 없음. 다만 "실효적으로 `floor(α·l)`"이라는 규칙이 어디에도 명시돼 있지 않음 |
| 빈 출력 | **FAIL** | 전부 빈 출력이면 `Peak=1.0`. 발견 #9 |
| 길이가 서로 다른 출력 | **PASS** | 가장 긴 것 하나가 `l`을 정함. 테스트로 고정됨 |

### 3.3 로컬 검증 재실행

| 명령 | 결과 |
|---|---|
| `pipeline/.venv/bin/pytest -q -rs` | **PASS — 415 passed, 2 skipped, 36.97s** |
| skip 사유 | `tests/test_loader_bf16.py:5: could not import 'torch'`, `tests/test_real_model_adapter.py:16: could not import 'torch'` — 로컬에 PyTorch 미설치. 실패가 아니라 GPU 어댑터 테스트 생략 |
| 베이스라인 대조 | `git archive HEAD`를 스크래치패드에 풀어 실행: **413 passed, 2 skipped**. 원래 빨간 테스트는 없고, 작업 트리 변경은 테스트 2개를 순수 추가 |
| 함의 | HEAD의 테스트 전부가 길이 100 시퀀스만 써서 **짧은 출력에서의 `l` 의미를 고정하지 않았다.** 두 정의 모두 HEAD 테스트를 통과한다 — 이것이 코드/문서 불일치가 지금까지 안 잡힌 이유다 |
| `pipeline/scripts/run_dry_run.py` | **PASS, exit 0.** invariant 4종 전부 통과 (`logodds_matches_paper_table`, `pooling_guard_fires`, `contaminated_scores_higher_than_clean`, `contaminated_has_higher_partial_pass`) |
| `git diff --check` | **PASS**, rc=0 |

드라이런이 출력한 d / AUC / 상관 / 기저율 값은 합성 진단이며 연구 결과로 읽지 않았고, 이 문서 어디에도 옮기지 않았다.

### 3.4 분석 계층 대 논문 프로토콜

| 항목 | 판정 | 근거 |
|---|---|---|
| C1–C3이 Qwen2.5-32B에 고정 | **PASS** | `run_analysis.py:90` `CONFIRMATORY_MODEL = QWEN2_5_32B.name` — CLI 플래그 아님 |
| bf16 → BNB-nf4 고정 | **PASS** | `run_analysis.py:91-92`, 역시 모듈 상수 |
| LCB 전체 문항, 공통 complete-case 집합 | **PASS** | `study_inputs.py:205-256` — 6개 점수 중 하나라도 없으면 세 배열에서 **함께** 제거, `n_items_dropped_incomplete`로 계수 |
| C4가 690 / 182 대리 집단 | **PASS (구조)** | `c4_inputs`가 `primary_label`에서 유도하고 `n_possible_exposure` / `n_shared_clean_control`을 산출물에 기록. 실제 690/182는 본 실행 데이터가 있어야 확인 — 현재 **UNVERIFIED** |
| 여섯 AUC의 결합 DeLong 공분산 | **PASS** | `confirmatory.py:381-394` — 3탐지기 × 2정밀도를 한 `delong_auc_covariance` 호출에 넣고, gap SE는 가중치 (+0.5, +0.5, −1) 선형 대비에서 산출 |
| 추정 불가 slot이 `p=1`로 남는지 | **PASS** | `paired_score_shift_test`의 `not_estimable()`이 `p_value=1.0`, `c4_rank_reversal_test`도 동일. `assemble_confirmatory_family`가 네 slot 누락을 예외로 거부 |
| 네 slot Holm 보정 | **PASS** | `holm_adjusted_p_values`, `statsmodels` 대조 테스트 존재 |
| Q2 구간이 per-model item-stratified conditional logistic Wald | **PASS** | `run_analysis.py:170-204` → `fit_conditional_logit_by_model`, `interval_method="item_stratified_conditional_logit_wald"` |
| 변분 Bayes 사후 SD를 구간으로 쓰지 않는지 | **PASS** | `run_analysis.py`는 `mixed_effects.fit_vb`를 호출하지 않음. VB는 `coverage.py`의 비교 대상으로만 등장하고 분석 manifest에 `ruled_out`으로 명시 |
| validation output을 analysis가 거부 | **PASS** | `require_main_study()`가 파케이 파일 하나 열기 전 첫 호출 (`run_analysis.py:377`). `tests/test_study_phase.py` 존재 |
| interval coverage record 존재 | **PASS** | `analysis_artifacts/interval_coverage_check.json`. conditional logit 0.950 (190/200), VB 0.485 (97/200), 명목 0.95. 시나리오 2개(β_QE=0.0, 0.5), 각 200회, seed 20250918 |
| 생성 조건·결과가 분석 manifest에 연결되는지 | **PASS** | `build_analysis_manifest`가 sha256, `generating_parameters`, `n_replications`, `seed`, `achieved_coverage`, `regenerate_with`를 전부 기록 |
| coverage record 자체에 git commit이 있는지 | **FAIL** | record에 `git_commit`도 `study_phase`도 없음. 분석 manifest가 sha256으로만 묶음 |
| 코퍼스 증거가 Q1b 라벨로 역주입되지 않는지 | **PASS** | `temporal_labels.py`와 `analysis/*.py` 어디에도 `corpus_reference` / `tracer` 참조 없음 (grep 0건) |

### 3.5 H100 실행 전 게이트

접속 정보도 권한도 없어 **H100에는 접속하지 않았다.** 아래는 전부 실제 상태를 확인하지 않은 항목이다.

| 게이트 | 판정 |
|---|---|
| 정확한 commit checkout + clean worktree | **FAIL** (현재 dirty, 게이트도 없음) |
| pinned H100 환경 설치와 `pip freeze` | **UNVERIFIED** — `requirements-h100.txt`와 `envs/local-smoke-freeze.txt`는 존재하나 현재 H100 상태 미확인 |
| Llama gated-model 접근권한 | **UNVERIFIED** |
| 다섯 모델 immutable revision 해석 | **PASS (기록) / UNVERIFIED (해석)** — `registry.py`에 다섯 개 모두 SHA 고정. Hub에서 실제로 해석되는지는 네트워크 확인 안 함 |
| 다섯 AWQ checkpoint 존재 | **UNVERIFIED** (로컬엔 없음) |
| 각 `quantization_manifest.json` | **UNVERIFIED** |
| 각 `calibration_overlap_report.json` | **UNVERIFIED** |
| overlap 발견 시 판정·기록 절차 | **PASS (구현)** — `require_calibration_overlap_report`가 보고서 부재만 차단하고, 0이 아닌 overlap은 기록 후 판정 대상으로 남김 |
| Llama-3.1-8B bf16/BNB/AWQ 경로 | **UNVERIFIED** (한 번도 실행 안 됨) |
| Qwen2.5-32B / Olmo3.1-32B bf16 메모리 | **UNVERIFIED** — 논문 §4.3 표의 ~64–65 GB는 파라미터 수 유도값이며 측정값이 아니라고 §4.3이 스스로 명시 |
| 두 32B의 BNB-nf4 / AWQ 메모리와 생성 경로 | **UNVERIFIED** — 특히 AWQ는 발견 #5 |
| 유한 fixed-prompt log probability | **PASS (7B/nf4) / UNVERIFIED (나머지 arm)** |
| LCB stdin/functional 코드 추출 | **PASS (7B/nf4) / UNVERIFIED (나머지 arm)** — `run_lcb_smoke_test.py` |
| raw parquet schema와 manifest | **PASS** — 테스트 통과, 스모크에서 mock과 스키마 일치 확인 기록 |
| 중단 후 resume 동작 | **FAIL (테스트 부재)** — 코드에는 생성 캐시 + `config_hash` 거부가 있으나 테스트가 없음 |
| validation / main-study namespace 분리 | **PASS** — manifest 필수 필드 + `require_main_study`, 테스트 통과 |

---

## 4. 최소 수정안

세 건 모두 **논문의 연구 의미를 바꾸지 않는다.** 1번은 논문이 이미 기술한 것을 코드가 따르게 만드는 수정이고, 2·3번은 기록·게이트만 추가한다. 논문 §4.6이 "채점 하네스의 결함 수정"과 "런타임·메모리 실패로 인한 운영 변경(배치 크기 등)"을 결과 열람 **전에** 한해 명시적으로 허용하므로, 셋 다 그 범위 안에 있다. 현재 관측 데이터는 0건이다.

### 수정 1 — CDD 정의 커밋 (P0-1)

- **대상**: `pipeline/src/qcd/detectors/cdd.py`, `pipeline/tests/test_detectors.py` (이미 작업 트리에 있음)
- **내용**: 코드 변경 없이 그대로 커밋
- **추가 권장**: `pipeline_implementation_log.md:107-109`의 `α·l_max` 서술 동기화
- **테스트**: 현재 415 passed로 이미 통과
- **논문 의미 변화**: 없음 — `paper_draft.md:683-686`이 이미 이 정의를 쓴다

### 수정 2 — dirty 작업 트리 기록 (P0-2)

**최소안**: `io/manifest.py`에 두 필드 추가.

- 대상: `pipeline/src/qcd/io/manifest.py`
- 내용: `get_git_commit_hash` 옆에 `git status --porcelain` 기반 `git_dirty: bool`과 `git diff HEAD`의 SHA-256 `git_tracked_diff_sha256: str | None`을 추가하고 `RunManifest`에 싣는다. `config`가 아니라 **최상위 필드**여야 한다 — `config`에 넣으면 dirty 상태가 `config_hash`를 바꿔 정상 resume을 깨뜨린다.
- 테스트: dirty/clean 양쪽에서 필드가 채워지는지 1건.
- 논문 의미 변화: 없음.

**강화안**: 위에 더해 `run_main.py`가 dirty면 **실행 자체를 거부**.

- 대상: `pipeline/scripts/run_main.py`
- 내용: `study_phase=MAIN_STUDY`일 때 dirty면 `SystemExit`. `--allow-dirty` 탈출구를 둘지는 사용자 결정 사항. 스모크·드라이런은 영향 없음.
- 테스트: 거부 경로 1건 + `config_hash` 불일치 거부 경로 1건(현재 미커버).
- 논문 의미 변화: 없음. §4.6의 "변경 후 동결" 요구를 코드로 강제하는 것.

`CDD_SCORE_DEFINITION` 버전 문자열만으로는 부족하다는 점이 이번에 실증됐다 — 커밋 `9abd8b5`가 문자열을 `v2`로 올린 뒤에도 코드는 v1 동작을 유지했고, `config_hash`는 그 차이를 전혀 보지 못했다.

### 수정 3 — 처리량 (P0-3)

측정이 먼저다. 코드 수정안을 지금 제시하지 않는다. 5절의 측정을 받은 뒤에 배치 여부를 결정해야 하며, 배치는 §4.4가 고정한 seed 정책(`sha256(item_id, sample_id, temperature)`를 각 생성 직전에 설정)과 상호작용하므로 seed 재현성을 깨지 않는 설계인지 별도 확인이 필요하다.

---

## 5. 실행 비용 검산 (독립 계산)

기준: LCB 1,055 + HumanEval 164 + MBPP+ 378 = **1,597 문항/모델/정밀도**

| 항목 | 값 | 계산 |
|---|---|---|
| arm-item 조건 수 | **31,940** | 1,597 × 5모델 × 4정밀도 |
| 총 생성 수 | **1,628,940** | 31,940 × 51 (greedy 1 + 샘플 50) |
| 그중 greedy | 31,940 | |
| 그중 CDD 샘플 | 1,597,000 | |
| 고정 프롬프트 채점 forward pass | 31,940 | 조건당 1회 |
| 샌드박스 실행 | 31,940 | greedy만 채점 |
| 7·8B 세 모델 몫 | 977,364 생성 | 3 × 4 × 1,597 × 51 |
| 32B 두 모델 몫 | 651,576 생성 | 2 × 4 × 1,597 × 51 |

**기존 스모크 처리량을 적용할 수 있는 범위.** 저장소에 기록된 유일한 실측은 `pipeline/README.md`의 **Qwen2.5-7B-Instruct / BNB-nf4 / HumanEval 최단 프롬프트 5문항 / 문항당 3생성(1 greedy + 2 샘플) → 문항당 10–19초**다. 생성당 3.3–6.3초에 해당한다. 이 수치가 적용 가능한 범위는 **7B, nf4, 짧은 HumanEval 프롬프트, 배치 1**뿐이다. 32B에 그대로 일반화하지 않는다. LCB 프롬프트는 `question_content` 전체라 훨씬 길고, 생성 길이 분포도 다르다.

그 범위 안에서만 환산하면: 1,628,940 × 3.3–6.3초 ≈ **1,490–2,850 GPU-시간**. 7B 수치를 그대로 쓴 값이므로 32B 두 arm을 포함한 실제 총량은 이보다 크다. 단일 80GB H100에서 이 계획은 완주할 수 없다.

**저장 공간.** 조건당 51행 × 1,628,940행. 행마다 `token_ids`(int64 배열), `token_logprobs`(float64 배열), 생성 텍스트가 들어간다. 생성 길이 평균 L에 대해 행당 대략 16L 바이트 + 텍스트. L=300이면 파케이 압축 전 약 10 GB, 압축 후 5–7 GB로 추정된다. 여기에 `GenerationCache`가 같은 페이로드를 pickle로 **한 벌 더** 저장하므로(`cache.py:60-64`, 2단 fanout, 파일 163만 개) 총량은 대략 두 배다. L이 미측정이라 이 추정의 오차는 크다.

**결정에 필요한 추가 측정** (스모크 범위, 결과값 비교 없음):

1. **조건별 생성 길이 분포.** 모델·정밀도별 평균/중앙값/512 cap 도달률. GPU 시간과 저장 공간의 지배 변수인데 현재 단 하나도 측정돼 있지 않다. `truncated_at_cap`이 이미 저장되므로 스모크 확장만으로 얻을 수 있다.
2. **32B 네 정밀도의 실제 peak 메모리와 생성당 초.** 특히 bf16(~65 GB 추정)과 AWQ(7B 실측 근거로 bf16 폭 추정). 80 GB 카드에서 KV 캐시 여유가 실제로 얼마나 남는지.
3. **LCB 프롬프트에서의 생성당 초.** HumanEval 최단 프롬프트 5개는 대리값이 되지 못한다.
4. **재시작 단위.** 현재 `run_main.py`는 모델 → 정밀도 → 문항 순 단일 루프이고 재시작은 전체 재순회 + 캐시 히트에 의존한다. 위 1–3을 받은 뒤에 (모델, 정밀도) 셀 단위 분할 실행이 필요한지 판단해야 한다.

---

## 6. Olmo / TRACER 작업 분리

### (a) GPU 본 채점 시작 전에 반드시 필요 — 없음

논문 §5가 step 5(TRACER)를 step 6과 병행 가능으로 두고, 코퍼스 증거는 시간 라벨과 분리된 축에 들어간다. 코드에서도 `temporal_labels.py`와 분석 모듈이 코퍼스 증거를 전혀 참조하지 않는 것을 확인했다. 따라서 이 갈래는 본 채점을 막지 않는다.

### (b) 병행 가능하지만 논문 완성 전 필요

| 항목 | 상태 |
|---|---|
| Olmo3-7B와 Olmo3.1-32B 코퍼스를 별개 처리 | **구현됨.** `olmo_corpora.py`가 checkpoint↔repository 짝을 강제하고 반대 arm 짝을 거부. 스캔 진입점마다 `--model` 필수, 증거 행마다 기록. manifest 스키마 v4가 `model`을 담고, v3 manifest는 `finalize`가 거부 |
| 32B pretraining corpus revision | **미고정.** `allenai/dolma3_mix-6T`에 스캔 revision 없음. `require_revision`이 거부하므로 조용히 통과하지는 않음 |
| Dolci SFT/DPO/RL 체크포인트 귀속 | **미검증.** 세 repository 모두 "assignment not verified". 스크립트가 operator-declared임을 stderr로 출력 |
| n-gram / candidate bound 설정 | **미동결.** `n=13`, coverage `0.8`, `--candidates-per-item 5` 전부 provisional |
| `no-match-found`가 clean으로 바뀌지 않는지 | **PASS.** 3상태 enum(`confirmed-match` / `no-match-found` / `not-observable`). `no-match-found`는 완전 스캔 후에만 발행되고, bounded 스캔은 `not-observable`. 두 문서가 "음성 정답 아님"을 반복 명시 |
| TRACER 결과가 Q1b 라벨로 역주입되지 않는지 | **PASS.** grep 0건 + `TRACER_REIMPLEMENTATION.md`가 명시적으로 금지 |

### (c) 현재 미구현 또는 미동결

| 항목 | 상태 |
|---|---|
| AST / 편집거리 계열 (계열 ii) | **미구현.** `OLMO_GROUND_TRUTH.md`의 "Not implemented in this first pass"에 명시 |
| embedding / LLM 패러프레이즈 계열 (계열 iii) 운영 구현 | **부분.** `tracer_embedding.py`에 어댑터는 있으나 후보 검색·판정 배선은 없음 |
| Jina revision과 embedding 설정 | **부분 고정.** `jinaai/jina-embeddings-v3`를 immutable commit으로 고정, `text-matching` task, L2 정규화 코사인, 음수 클리핑. 전부 재구현 선택이며 원 논문 설정이 아니라고 문서가 명시 |
| LLM normalizer / verifier / screener identity와 디코딩 | **미동결.** `TRACER_REIMPLEMENTATION.md`가 체크포인트, system prompt, temperature, top-p, seed, 출력 토큰 상한, 재시도 정책을 "remain to be frozen"으로 열거 |
| confirmed positives 대상 validation | **미실행.** 재현율 측정, 층화 표본 특이도, 정규화 의미 보존 감사 전부 남아 있음 |
| 인덱스 서비스 | **없음.** 3–4 TB 압축 코퍼스 전량 전송이 현재의 fallback |

---

## 7. 사용자 승인을 요청할 항목

감사 시점까지 아무것도 실행하지 않았다. 아래는 각각 별도 승인이 필요하다.

| # | 요청 | 범위 |
|---|---|---|
| 1 | **코드 수정 — CDD 커밋** | 작업 트리의 두 파일을 그대로 커밋. 내용 변경 없음 |
| 2 | **코드 수정 — manifest dirty 기록 (최소안)** | `io/manifest.py`에 `git_dirty` + `git_tracked_diff_sha256` 추가, 테스트 1건 |
| 3 | **코드 수정 — `run_main.py` dirty 거부 (강화안)** | 2번을 채택할 때만 의미 있음. `--allow-dirty` 탈출구 유무를 지정해야 함 |
| 4 | **코드 수정 — resume / config_hash 거부 테스트 추가** | 순수 테스트 추가 |
| 5 | **문서 수정** | `pipeline_implementation_log.md`의 `α·l_max` 동기화. 채택 시 영어 정본과 한국어 미러를 같은 작업에서 맞춘다 (이번 항목은 두 드래프트를 건드리지 않는다 — 이미 정확하다) |
| 6 | **HF 메타데이터 조회 (다운로드 없음)** | 다섯 모델 revision SHA가 실제로 해석되는지, Llama gated 접근 여부를 Hub API로만 확인. 가중치는 받지 않음 |
| 7 | **H100 접속 및 스모크 테스트** | 접속 정보 필요. 범위: 세 미검증 arm × 네 정밀도의 로딩·메모리·유한성·스키마·처리량·생성 길이 분포. 결과값의 방향·크기는 비교하지 않음 |
| 8 | **AWQ 생성** | 다섯 모델. 6·7번 이후 |
| 9 | **대규모 코퍼스 스캔** | Dolci 3종(약 4.35 GB) 먼저, 사전학습 mix는 32B revision 고정 후 |
| 10 | **커밋 여부** | 1–5번을 승인하더라도 커밋은 별도 지시가 있을 때만 한다 |

---

## 부록 — 요청 범위 밖 관찰 1건

CDD는 100토큰까지만 쓰는데 샘플 50개는 512토큰 상한까지 생성된다. §4.4가 512 cap과 n=50을 고정했으므로 바꿀 사안이 아니지만, 5절의 GPU 시간 대부분이 여기서 나온다.
