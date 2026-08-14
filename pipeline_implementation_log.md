# 파이프라인 구현 기록 (2026-08-14)

**대상:** `pipeline/src/qcd/` — mock 검증 가능 범위 전체
**환경:** 맥(Apple Silicon, CUDA 없음). `pipeline_build_plan.md`가 상정한 RTX 4060 랩톱도,
H100 박스도 이 세션에서는 접근 불가. 사용자와 사전 합의한 범위: **mock 프로파일 + 실제
네트워크/CPU 전용 코드**(데이터셋 다운로드, 샌드박스 코드 실행, 순수 통계)까지 전부 짜고,
GPU 실경로(`models/loader.py`의 real `generate()`/`score_logprobs()`, GPTQ/AWQ 백엔드,
`scripts/run_smoke_test.py`)는 기존 `NotImplementedError` 스텁 그대로 남겨둠.
**결과: 신규 모듈 24개, 신규 테스트 파일 15개, 스크립트 4개, 테스트 4개 → 119개로 증가, 전부 통과.**

---

## 1. 데이터 로더 (`data/`)

### 1.1 `data/livecodebench.py`
**문제 발견:** 설치된 `datasets==5.0.1`이 스크립트 기반 데이터셋 로딩을 완전히 제거해서
`datasets.load_dataset("livecodebench/code_generation_lite", ...)`가
`RuntimeError: Dataset scripts are no longer supported`로 즉시 실패한다. 확인 후 우회
전략으로 전환: HF Hub의 원본 로딩 스크립트(`code_generation_lite.py`, 지금은 죽은 코드)가
가진 `ALLOWED_FILES` 매니페스트를 그대로 가져와 `_RELEASE_FILES`에 하드코딩하고,
`huggingface_hub.hf_hub_download`로 `test{,2..6}.jsonl` 원본 파일을 직접 받는다.

**검증한 사실 (재유도 아님, 실측):**
- 6개 파일은 `question_id` 기준으로 겹치지 않는 청크이며 `contest_date` 오름차순
  (`test.jsonl`: 2023-05-07~2024-03-02 ... `test6.jsonl`: ~2025-04-06) — CLAUDE.md §4.2의
  "LCB 수집은 2023-05부터 시작"과 일치.
- `release_v6` 총 1,055문항 (400+111+101+101+167+175).
- 문항의 `metadata` 필드(JSON 문자열)가 `func_name`을 담고 있음 — functional 테스트타입에
  필요, 처음 구현에서 누락했다가 스코어링 단계 설계 중 발견해 추가.

**핵심 함수:** `load_livecodebench_split(cutoff_boundary, release_version) -> (pre_items, post_items)`
— `cutoff_boundary`는 호출자가 넘기는 파라미터로, 논문 §4.2의 arm별 사전 지정 경계 규칙을
이 모듈이 하드코딩하지 않도록 설계.

### 1.2 `data/humaneval.py`, `data/mbppplus.py`
`evalplus.data.get_human_eval_plus`/`get_mbpp_plus`의 얇은 래퍼. 실측으로 **정확히 164개
/ 378개**임을 확인(`pipeline_build_plan.md`의 "open assumption #3" 해결) — 개수가 어긋나면
조용히 넘어가지 않고 `RuntimeError`. `evalplus==0.3.1`을 `requirements-local.txt`에 고정.
각 문항의 전체 evalplus problem dict(`canonical_solution`, `base_input`, `plus_input`,
`atol`, `entry_point`)는 `Item.metadata["evalplus_problem"]`에 원형 그대로 보관 —
`scoring/sandbox.py`가 evalplus 자체 함수에 그대로 넘길 수 있도록.

---

## 2. 생성 (`generation/`)

### 2.1 `generation/cache.py`
문항·모델·정밀도·샘플 단위로 콘텐츠 주소화된 캐시(`CacheKey.digest` = 필드 전체 + 프롬프트
텍스트의 sha256). §4.4의 "1·2단계가 같은 생성물을 공유해야 한다"는 비용 절감 지시를 코드로
강제하기 위한 계층. 로컬 파일시스템에 2단계 해시 팬아웃(`cache_dir/xx/<digest>.pkl`)으로
저장 — 대량 실행 시 한 디렉터리에 파일이 몰리는 것을 방지.

### 2.2 `generation/sampler.py`
`sample_item(model, cache, ...) -> ItemGenerations(greedy, samples)` — greedy 1개 +
`CDD_N_SAMPLES`(=50)개의 T=0.8 샘플. `models.loader.LoadedModel` 프로토콜만 알면 되므로
mock이든 실제 백엔드든 동일 코드로 동작. 캐시 히트 시 `model.generate()`를 아예 호출하지
않음 — 테스트에서 캐시 두 번째 호출 시 모델을 아예 등록 해제한 뒤에도 값이 나오는지로
검증.

---

## 3. 채점 (`scoring/`)

### 3.1 `scoring/sandbox.py` — 가장 손이 많이 간 모듈
**HumanEval+/MBPP+:** 새로 만들지 않고 `evalplus.eval.untrusted_check` +
`evalplus.gen.util.trusted_exec`를 그대로 재사용 (`evalplus.evaluate.get_groundtruth`가
쓰는 것과 동일한 레시피 — canonical_solution을 먼저 돌려 기댓값·기준 시간을 구하고, 그걸
후보 코드 채점에 넘김). `evalplus`가 이미 `reliability_guard`/`time_limit`/`swallow_io`로
격리해주므로 직접 샌드박스를 재구현하지 않음.

**부분 점수 계산의 함정:** `untrusted_check`가 반환하는 `details` 배열은 크래시/타임아웃으로
중단되면 **시도한 만큼만** 잘려서 돌아온다(`details[:progress.value]`). 분모를
`len(details)`로 두면 못 돌려본 테스트가 "제외"되어 점수가 부풀려진다 — 반드시
`len(base_input)+len(plus_input)`(전체 테스트 수)를 분모로 써야 "부분 통과율"이 의미를
가짐. 이 함정을 문서화하고 코드에 명시.

**LiveCodeBench:** evalplus 형식이 아니라서 커스텀 subprocess 하네스 2종을 새로 작성:
- **stdin형** (Codeforces/AtCoder): 표준입력으로 넣고 표준출력을 줄 단위로 비교 (trailing
  공백/빈 줄 정규화만, 부동소수점 허용오차는 없음 — 한계로 명시).
- **functional형** (LeetCode): `metadata`의 `func_name`으로 `Solution` 클래스 메서드를
  호출. `input` 필드가 **줄바꿈으로 구분된 JSON 인자들**(다중 파라미터 함수는 한 줄에 한
  인자)이라는 걸 실제 LCB 문항 8개를 파라미터 수 다양하게 뽑아 직접 확인 후 파싱 로직 작성
  — 추측이 아니라 실측.

**`private_test_cases` 디코딩:** `base64(zlib.compress(pickle.dumps(json_string)))` —
이것도 실제 문항 데이터를 단계별로 까보며 확인한 인코딩. `pickle.loads`는 신뢰 경계에
민감한 연산이라 독스트링에 "LiveCodeBench 공식 HF 데이터셋 하나에만 적용, 임의의 네트워크
입력이 아님"이라고 명시.

**macOS 버그 발견·수정:** evalplus의 `reliability_guard()`가 `resource.setrlimit(RLIMIT_AS,
...)`를 호출하는데 macOS에서 `ValueError: current limit exceeds maximum limit`로 즉시
죽는다(리눅스와 RLIMIT_AS 처리 방식이 다름). evalplus가 제공하는
`EVALPLUS_MAX_MEMORY_BYTES=-1` 탈출구를 macOS에서만, 사용자가 이미 설정하지 않은 경우에만
기본값으로 넣도록 수정 — H100/Linux에서는 실제 메모리 캡이 그대로 유지됨.

### 3.2 `scoring/pass_rate.py`, `scoring/logprob.py`
`pass_rate.py`는 `Item.dataset`에 따라 evalplus 경로/LCB 경로로 디스패치하는 얇은 파사드.
`logprob.py`는 `LoadedModel.score_logprobs()`의 1줄짜리 래퍼 — Min-k%가 전체 토큰별
로그확률 배열을 필요로 하므로(요약 스칼라 아님) 이걸 읽는 지점을 하나로 모으는 용도.

---

## 4. 탐지기 (`detectors/`)

### 4.1 `detectors/cdd.py`
`constants.py`에 이미 박혀 있던 공식을 그대로 구현: `Peak(M;x) = (1/n) Σ I(ED(s_i,
s_greedy) ≤ α·l_max)`. 토큰 단위 Levenshtein distance를 O(n·m) DP로 직접 구현(scipy 없음
컨벤션), star topology(모든 샘플을 greedy와만 비교, 샘플끼리는 비교 안 함), `l_max=100`
토큰으로 자른 뒤 계산. 임계값이 정확히 `α·l_max=5.0`이 되는 경계값 테스트까지 포함.

### 4.2 `detectors/perplexity.py`
표준 정의(평균 음의 로그확률의 지수). AUC 계산에 실제로 넣는 건 `perplexity()` 자체가
아니라 `negative_log_perplexity_score()`(부호 뒤집은 평균 로그확률) — "점수가 높을수록
오염 의심"이라는 세 탐지기 공통 방향을 맞추기 위함.

### 4.3 `detectors/mink_prob.py` — 원문 대조 필수 항목
CLAUDE.md의 "공식을 추측하지 말고 원문과 대조하라" 규율을 그대로 적용. Shi et al. 2023
(*Detecting Pretraining Data from Large Language Models*, arXiv:2310.16789, ICLR 2024)을
검색 → PDF를 `pdfs/2310.16789.pdf`로 새로 받음 → pdfplumber로 직접 추출한 원문에서 수식을
확인:

```
Min-K%(x) = 최저 확률 토큰 k%의 집합
MIN-K%PROB(x) = (1/E) Σ_{x_i∈Min-K%(x)} log p(x_i|x_1,...,x_{i-1})
```

"If the average log likelihood is high, the text is likely in the pretraining data"까지
원문에서 확인 — 점수 방향(높을수록 멤버/오염)도 추측이 아니라 인용. `k=20`이 논문 자체의
스윕(10/20/30/40/50) 최적값이라는 것도 확인 후 기본값으로 채택.

### 4.4 `detectors/threshold.py`
`constants.CDD_XI_FIXED=0.01`(원 논문 고정 임계값)을 쓰는 `classify()`와, held-out
칼리브레이션에서만 써야 하는 `select_threshold_youden()`을 분리. 후자는
`calibration_item_ids`/`evaluation_item_ids`를 받아 **겹치면 예외를 던지는 구조적 가드**를
넣음 — arXiv:2603.03203이 스스로 경고한 "gives CDD every advantage" 낙관 편향(평가셋에서
Youden index 재선택)을 코드 수준에서 재현 불가능하게 막음.

---

## 5. 분석 (`analysis/`)

### 5.1 `analysis/logodds.py`
§4.5.3의 기저율 교란 모형을 그대로 재구현: `sigmoid(alpha + sigma*Z - beta)`를 `Z~N(0,1)`에
대해 수치 적분으로 주변화(scipy 없이, numpy 배열 + 수동 사다리꼴 공식 — `np.trapz`/
`np.trapezoid` 이름이 numpy 버전 간 바뀌는 문제를 피하려 직접 구현). 8차 검토에서 이미
독립 재검증했던 표(β=0.50 → 5.6pp/7.7pp/−2.2pp 등)를 이번엔 재사용 가능한 함수로 승격.

### 5.2 `analysis/aggregation.py`
"HumanEval과 MBPP+를 풀링하지 말 것"(CLAUDE.md §5 5번)을 코드로 강제하는 가드.
`assert_not_pooled(datasets)`가 두 조건이 동시에 있으면 `PooledSecondaryConditionsError`를
던짐. LCB pre/post는 애초에 서로 비교하려고 만든 축이므로 허용.

### 5.3 `analysis/power.py`
논문의 검정력 표들을 파일럿 실측값(d, r, e)을 인자로 받는 함수로 일반화.

**여기서 진짜 버그를 잡음:** `items_needed_with_label_noise`의 첫 버전이 라벨 오차율 e에
따라 목표 ΔAUC만 감쇠시키고 `auc` 파라미터는 감쇠 안 된 원래 0.70을 그대로 넘기고
있었다 — 이건 CLAUDE.md §4.1이 명시적으로 "채택하지 않는다"고 적어둔 **"SE 고정"** 열을
재현하는 코드였다(결과: 266/472/1060). 실제 정본이 채택한 **"SE도 감쇠"** 열
(287/541/1268)을 재현하려면 관측 AUC 자체도
`AUC_obs = 0.5 + (1-2e)(AUC_true-0.5)`로 감쇠시켜 그 지점에서 SE를 평가해야 한다. 발견 →
수정 → 회귀 테스트(`test_items_needed_with_label_noise_requires_se_attenuation_not_just_delta`)로
두 값이 다르다는 것 자체를 고정.

### 5.4 `analysis/mixed_effects.py` — 의도적 예외 사항
논문 §4.5.5의 `correct ~ precision*contaminated + (1|item) + (1|model)` 교차 랜덤효과
로지스틱 GLMM. CLAUDE.md의 "scipy 없음" 규율은 문서 자체 문맥상 **"논문 자신의 검정력·AUC
표를 손으로 재현할 때"**로 범위가 좁혀져 있음(검증 가능한 목표 숫자가 있는 계산). GLMM은
비교할 손 계산 정답이 없는 실측 데이터 분석이라 다른 종류의 계산이고, 이걸 이분탐색으로
직접 구현하는 게 오히려 그 규율이 경계하는 "검증 안 된 수치 코드"가 된다고 판단해
`statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM`(변분 베이즈)을 채택 —
`requirements-local.txt`에 이 모듈 전용으로 추가, 근거를 모듈 독스트링에 명시.

교차효과 항의 정확한 파라미터 이름은 patsy의 카테고리 기준 레벨 선택에 따라 데이터마다
달라지므로(`"precision[T.quant]:contaminated[T.True]"` 등), 이름에 `":"`가 포함된
고정효과를 찾아 프로그램적으로 뽑아냄 — 이름을 하드코딩하지 않음. 테스트는 알려진 참
계수로 시뮬레이션한 데이터를 적합시켜 회수 범위를 확인하는 방식(정확한 자릿수 재현이
아니라 부호·근사 크기 확인 — GLMM은 손 계산 정답이 없다는 점을 테스트 자체에 명시).

---

## 6. 파일럿 (`pilot/`)

### 6.1 `pilot/cdd_gate.py`
§4.6의 go/no-go 관문. `check_cdd_gate(measured_baseline_auc)`가
`constants.CDD_GATE_AUC`(0.7936) 이상인지 보고 `CDDGateResult(passed, reason, ...)`를
반환. 통과/실패 각각의 사유 문자열에 "Contribution 4로 보고" 등 논문 §1의 후속 조치까지
박아둠.

### 6.2 `pilot/pilot_report.py`
§4.7의 다섯 가지 파일럿 수량 (a)~(e) 각각의 통계 원시 함수: `cohens_d_paired`(대응
Cohen's d), `pearson_r`, `base_rate`, `proxy_label_error_rate`(Olmo3 arm 라벨 오차율 e).
원시 데이터 형식을 모르는 채로 설계 — 실제로 붙이는 건 `dry_run.py`/`real_run.py` 쪽 책임으로
분리.

---

## 7. 입출력 (`io/`)

### 7.1 `io/raw_writer.py`
3-테이블 parquet 스키마: `items.parquet`(문항 메타), `generations.parquet`(문항 단위 생성
결과), `detector_scores.parquet`(탐지기 점수). `Item.metadata`는 데이터셋마다 키가 달라서
(LCB vs HumanEval+) pyarrow struct 컬럼으로 못 만들고 JSON 문자열 컬럼으로 저장.
"aggregate 저장은 대응·혼합효과 분석을 불가능하게 한다"는 §5 8단계 요구사항을 그대로
반영해 문항 단위 행만 씀.

### 7.2 `io/manifest.py`
git commit hash, config hash(JSON 정규화 후 sha256), 설치된 패키지 버전
(`importlib.metadata.version` — 실제 import 없이 조회되므로 torch가 없는 mock 프로파일에서도
안전하게 None 처리), hostname/platform, seed, timestamp를 묶은 `RunManifest`.

---

## 8. 오케스트레이션 (`dry_run.py`, `real_run.py`, `scripts/`)

### 8.1 `dry_run.py` (패키지 안에 위치, `scripts/`가 아님)
`scripts/run_dry_run.py`가 이 모듈의 `main()`을 부르는 얇은 CLI 셸일 뿐이고, 실제 로직은
설치된 `qcd` 패키지 안에 둬서 `tests/test_mock_pipeline_end_to_end.py`가 sys.path 조작
없이 바로 import할 수 있게 함(`pipeline_build_plan.md`의 파일 트리와 문자 그대로는 다르지만
"스크립트로도 pytest로도 둘 다 실행 가능"이라는 요구사항을 만족시키는 합리적 구현
디테일로 판단, 스크립트 파일 자체 독스트링에 이유를 남김).

4개 조건 × 4문항 = 16개 합성 문항을 만들고, `Item.contamination_proxy`를 그대로
MockModel의 숨은 정답으로 등록(`register_item`), fp16/bnb_nf4 두 정밀도에서 생성→채점
(mock의 `partial_pass_rate()`로 실제 샌드박스 우회 — mock이 만드는 토큰 스트림은 실행 가능한
파이썬이 아니므로 `pipeline_build_plan.md`가 이미 "mock bypasses this entirely"라고
명시한 부분)→3개 탐지기 채점→raw_writer 기록→pilot_report 다섯 수량 계산→CDD 관문
체크까지 끝까지 흘림. 마지막에 5가지 불변식을 검사:
1. 로그오즈 변환이 논문 표와 일치
2. HumanEval+MBPP+ 풀링 가드가 실제로 발동
3. CDD 관문 임계값이 `CDD_GATE_AUC` 상수 그대로(드리프트 없음)
4. 오염 문항이 청정 문항보다 세 탐지기 모두에서 높은 점수
5. 오염 문항이 청정 문항보다 부분 통과율이 높음

**여기서 진짜 버그를 하나 더 잡음:** 처음 돌렸을 때 `analysis.auc.quantization_delta_auc`가
`ValueError: p must be in (0, 1), got 1.0`로 죽었다. Mock의 오염/청정 분리가 소규모
데이터셋에서 너무 깨끗해서(AUC=1.0 정확히) `Φ⁻¹(1)`이 정의되지 않는 지점을 친 것. 이건 mock만의
문제가 아니라 **실제 파일럿에서도 작은 표본에서 완벽 분리(AUC=1.0)가 나올 수 있는 정당한
경계 상황**이라 판단해, mock 쪽을 손대는 대신 `auc.py`의 `quantization_delta_auc`
자체에 `auc = min(max(auc, 1e-9), 1-1e-9)` 클램프를 추가 — 관문 체크 함수가 실측 데이터에서도
안 죽도록 근본 수정.

### 8.2 `real_run.py`
`dry_run.py`와 같은 생성→채점→탐지→기록 루프를 실제(`mock=False`) 백엔드로 도는 공유
코어. `scripts/run_pilot.py`(PILOT_MODELS, bnb-nf4 우선)와 `scripts/run_main.py`
(MAIN_ANALYSIS_MODELS, 전체 양자화 사다리)가 이 위에서 모델·정밀도·문항 범위만 다르게
얹는 얇은 CLI. `_assemble_candidate_code()`가 HumanEval+/MBPP+는 `prompt+completion`,
LCB는 completion 단독이라는 두 데이터셋군의 차이를 처리(LCB의 마크다운 코드블록 추출은 아직
안 함 — 한계로 명시).

**테스트 설계에서 주의한 점:** `run()`이 `load_model(mock=False)`를 호출하는 지점에서 죽는
것을 확인하는 테스트를, 진짜로 `load_model`을 호출하게 두면 이 세션 환경(torch 미설치)에서
Qwen2.5-7B 토크나이저를 실제로 네트워크에서 받아버린 뒤 엉뚱한 에러(torch 없음)로 죽는다는
걸 미리 알아채고, `monkeypatch`로 `load_model`을 스텁으로 치환해 "배선이 정확히 그 지점까지
도달하고 그 지점에서만 멈춘다"를 네트워크 부작용 없이 검증하도록 수정.

### 8.3 `scripts/sync_from_h100.sh`
rsync 래퍼. `--exclude '*.safetensors' --exclude '*.bin'`으로 모델 가중치는 빼고
`data/raw/`만 동기화. 로컬 더미 디렉터리로 exclude 플래그가 실제로 작동하는지 확인(가짜
`.safetensors` 파일이 동기화되지 않는 것 확인), 인자 없이 실행 시 사용법 출력 후
exit 1 확인.

---

## 9. 기존 코드에서 발견해 고친 설계 결함

계획에는 없었지만 통합 과정에서 발견해 즉시 고친 것들 (전부 회귀 테스트 포함):

| 파일 | 문제 | 수정 |
|---|---|---|
| `models/loader.py`, `models/mock.py` | `LoadedModel.generate()`/`score_logprobs()`가 `contaminated: bool`을 인자로 받음 — 실제 모델이라면 자신이 평가받는 바로 그 정답 라벨을 생성 입력으로 받는 꼴 | `contaminated`를 공유 인터페이스에서 제거, `MockModel.register_item()`으로 사전 등록하는 방식으로 전환 |
| `analysis/power.py` | 라벨 잡음 보정 시 SE를 원래 AUC(0.70)에서 계산 — CLAUDE.md가 명시적으로 기각한 "SE 고정" 열을 재현 | 관측 AUC 자체를 `(1-2e)`로 감쇠시킨 뒤 그 지점에서 SE 평가 |
| `analysis/auc.py` | `quantization_delta_auc`가 AUC=1.0(완벽 분리)에서 `Φ⁻¹` 미정의로 크래시 | 입력을 `[1e-9, 1-1e-9]`로 클램프 |
| `scoring/sandbox.py` | evalplus의 메모리 가드가 macOS에서 `resource.setrlimit` 실패로 항상 죽음 | macOS에서만 `EVALPLUS_MAX_MEMORY_BYTES=-1` 기본 설정(사용자 설정 우선) |

---

## 10. 테스트 현황

```
119 passed (dataset loader 5, mock model 6, generation 5, scoring sandbox 8,
detectors 21, logodds 8, aggregation 6, power 11, mixed_effects 2, pilot 14,
io 9, dry-run e2e 5, real_run 5, 기존 schema/bootstrap/auc 관련 테스트 등)
```

전부 GPU 없이, mock 프로파일 또는 실제 네트워크(HF Hub/evalplus 다운로드)·실제 서브프로세스
코드 실행만으로 통과. `pytest`를 CI에 그대로 걸 수 있는 상태.

## 11. 남은 것 (다음 세션, CUDA 머신에서)

- `models/loader.py`의 `_RealModelAdapter.generate()`/`score_logprobs()` 실제 구현
- GPTQModel/llm-compressor 기반 GPTQ/AWQ 백엔드
- `scripts/run_smoke_test.py` (Qwen2.5-7B bnb-nf4 실제 로딩 스모크 테스트)
- `requirements-h100.txt`의 실제 driver/CUDA 버전 고정

`pipeline/README.md`의 "What's built vs. what's deferred" 절과 `pipeline_build_plan.md`
상단의 BUILD STATUS 배너에도 같은 내용을 남겨둠.
