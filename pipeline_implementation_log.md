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

---

# 실 GPU 경로 구현 및 검증 (2026-08-15)

**대상:** `pipeline/src/qcd/models/loader.py`의 `_RealModelAdapter.generate()`/
`score_logprobs()` (fp16/bnb-int8/bnb-nf4), `scripts/run_smoke_test.py` 신규 작성.
**환경:** 이번 세션은 실제 H100 80GB 박스(드라이버 580.126.09, CUDA 13.0, 툴킷 12.4) —
지난 세션(2026-08-14, macOS)이 CUDA 없어 스텁으로 남겨둔 지점을 여기서 이어받음.
사용자와 사전 합의한 범위: **fp16/bnb 실경로 + 스모크 테스트까지**만, GPTQ/AWQ 백엔드는
별도 세션으로 분리(모델별 GPTQ vs AWQ 선택·양자화 체크포인트가 미해결 open assumption이라
조사 성격의 작업이지 구현 작업이 아님).

## 1. 환경 설정

`pipeline/.venv` 생성 후 `requirements-local.txt` + `pip install -e .`, 이어서
`torch==2.6.0+cu124`(cu124 wheel index), `transformers==5.15.0`,
`bitsandbytes==0.50.1`, `accelerate==1.14.0` 설치. `HF_HOME`/`HF_HUB_CACHE`는
`/root/hf_cache`(로컬 오버레이 디스크, 1.6TB 여유) — 레포 밖, `.env.example` 지침대로.
Qwen2.5-7B는 비공개(gated)가 아니라 `HF_TOKEN` 없이 다운로드됨(Llama-3.1-8B-Instruct는
나중에 필요).

## 2. `models/loader.py` — `_RealModelAdapter` 구현

- `_RealGenerationSample`, `_seed_from()`을 이 모듈에 독립적으로 새로 둠 —
  `generation/sampler.py`의 `ItemGenerations.greedy`가 "untyped to avoid coupling
  to models.mock"이라고 이미 명시했던 설계 의도를 따라 mock.py의 동명 클래스/헬퍼를
  import하지 않음.
- `score_logprobs(item_id, token_ids)`는 프롬프트를 받지 않는 공유 Protocol 시그니처를
  그대로 유지하면서, `generate()` 호출 시 `self._prompts[item_id] = prompt`로 저장해두고
  나중에 조회하는 방식으로 해결. 등록 전에 호출되면 `RuntimeError`(mock의 `KeyError`
  등록-계약과 구분되도록 별도 예외 타입).
- `generate()`: `output_scores=True, return_dict_in_generate=True`로 생성 자체의 로짓을
  재사용해 토큰별 로그확률을 별도 forward pass 없이 계산. `torch.manual_seed(_seed_from(...))`
  로 재현성 확보.
- `score_logprobs()`: 저장된 프롬프트 + 전달받은 token_ids를 이어붙여 단일 forward pass로
  teacher-forced 로그확률 산출.
- `_load_bnb`의 nf4 분기에 `bnb_4bit_compute_dtype=torch.bfloat16` 추가 — 원래 없어서
  fp32 연산으로 떨어지던 것을 H100 bf16 텐서 코어를 쓰도록 수정(양자화된 가중치 자체는
  그대로 nf4, 연산 정밀도만 변경).

**실행 중 발견한 실제 버그:** `tokenizer.apply_chat_template(..., return_tensors="pt")`가
이 transformers 버전(5.15.0)에서 순수 텐서가 아니라 `BatchEncoding`(dict형, `input_ids`/
`attention_mask` 키)을 반환함. 이걸 그대로 `model.generate()`의 위치 인자로 넘기니
`inputs_tensor.shape[0]` 조회 시 `BatchEncoding.__getattr__`을 타고 들어가 의미 불명확한
`AttributeError`로 죽음(모델 로딩 384초 걸린 뒤라 재현 비용이 컸음 — 5-6분 다운로드 후
실패). 첫 스모크 테스트 실행에서 실제로 이 오류로 죽었고, `_build_input_ids()`가
`(input_ids, attention_mask)` 튜플을 명시적으로 언패킹해 반환하도록 수정하고
`model.generate()`/`model()` 양쪽에 `attention_mask`를 명시적으로 전달하도록 고쳐서
해결. 가중치가 캐시된 두 번째 실행에서 정상 통과.

## 3. `tests/test_real_model_adapter.py` (신규)

`hf-internal-testing/tiny-random-gpt2`(chat_template 없음 — plain-tokenization 폴백 경로
전용) 기준 CPU-only 회귀 테스트 5개: 생성 결과 token_ids/token_logprobs 길이 일치, greedy
결정성, `score_logprobs()` 유한값·길이 일치, 등록 전 호출 시 `RuntimeError`. `pytest.
importorskip("torch")`로 게이팅해 mock-only 프로파일에 영향 없음. `_RealModelAdapter`에
`max_new_tokens` 생성자 오버라이드를 추가해 이 테스트가 짧게 끝나도록 함(실제 7B 스모크
테스트의 기본값 512와 무관).

## 4. `scripts/run_smoke_test.py` (신규) — 실행 결과

Qwen2.5-7B-Instruct, BNB-nf4, HumanEval 최단 프롬프트 5개, greedy+T=0.8 샘플 2개.
실측값(H100, 가중치 캐시된 두 번째 실행 기준):

- 모델 로드: 10.7초 (첫 실행은 다운로드 포함 384초)
- 문항당: 10~19초 (greedy 236~512 토큰)
- **peak GPU 메모리: 6.69GB** (기대했던 ~4-6GB 대역과 부합, fp16 로드 시 예상되는
  15GB+와 확실히 구분됨 — nf4 양자화가 실제로 적용됐다는 증거)
- 체크리스트 8개 항목 전부 통과: 로그확률 유한, 반복 샘플이 실제로 다름, 샌드박스 pass
  rate가 [0,1] 범위, teacher-forced 재채점이 generate()와 길이·유한성 일치, 탐지기 3종
  전부 정상 범위, writer 스키마 정상, wall-clock 정상, `pip freeze` 저장.

## 5. 실행 중 발견한 별도 설계 결함 — 같은 세션에서 수정 완료

**HumanEval 실제 출력을 까본 결과, `partial_pass_rate`가 5문항 전부 0.0이었다.** 원인은
`real_run.py`의 `_assemble_candidate_code()`가 `item.prompt + completion_text`를 그대로
이어붙이는 방식인데, -Instruct 모델은 원시 코드 이어쓰기가 아니라 산문 설명 +
마크다운 코드펜스로 답한다(예: HumanEval/53 응답이 "Sure! The function `add` takes two
integer parameters..."로 시작하고 실제 코드는 ```python 블록 안에 있음). 기존 docstring은
이 문제를 LCB에만 국한해 명시했었는데("Extracting a code block out of a raw chat-style
model response... is a known gap, not handled here"), 실측 결과 -Instruct 로스터 전체를
쓰는 이상 HumanEval+/MBPP+에도 동일하게 적용된다 — 즉 두 조건의 pass rate는 현재
구조적으로 항상 과소추정(사실상 0)된다는 뜻. 샌드박스 실행 자체는 정상 동작하는 것으로
확인됐으므로(스모크 테스트 체크리스트 통과) 이건 스코어링 버그이지 실행 인프라 버그가
아님.

**사용자와 상의 후(프롬프트를 "이 코드를 완성하라"는 지시로 바꾸는 대안도 검토했으나
기각 — 벤치마크 프롬프트의 표면형을 바꾸면 Q1 탐지기가 근거로 삼는 "정확히 이 프롬프트에
대한 암기" 신호 자체가 흐트러질 위험이 있고, LCB는 애초에 코드-이어쓰기 프레이밍이 안 맞아
데이터셋별로 다른 프롬프트 템플릿이 필요해짐 — 대신 후처리 추출 방식으로 확정) 같은
세션에서 수정.** `real_run.py`의 `_assemble_candidate_code()`에 `_strip_markdown_fence()`
(마지막 ` ``` ` 펜스 블록 추출, 없으면 원문 그대로)를 먼저 적용한 뒤, evalplus 자체의
후처리(`evalplus.sanitize.sanitize`/`code_extract` — evalplus 리더보드가 LLM 출력을
채점하기 전에 쓰는 바로 그 모듈, `tree-sitter` 기반 AST 추출)를 재사용:

- **HumanEval+/MBPP+:** `sanitize(prompt + fenced, entrypoint=entry_point)` —
  `evalplus.sanitize.script()` 자신의 레시피 그대로. 목표 함수에 도달 가능한 정의만
  AST로 추출해 산문·불필요 코드를 버림.
- **LiveCodeBench:** `code_extract(fenced)` — entrypoint 없이. `sanitize()`의 AST 경로는
  import/class/function/assignment 노드만 보존하므로, 함수 래핑 없이 `input()`/`print()`를
  바로 쓰는 LCB stdin형 스크립트에서는 실행 로직이 통째로 날아갈 위험이 있음. `code_extract`
  는 그런 제약이 없어 더 안전한 선택. **이번 세션에서 실측 LCB 응답으로 검증하지는
  못함**(스모크 테스트가 HumanEval만 실행) — 추후 확인 필요.

**펜스를 먼저 벗겨내야 하는 이유(실측으로 발견):** `code_extract`만 단독으로 쓰면, 펜스
구분자 줄(` ```python `, ` ``` `) 자체가 유효한 파이썬이 아니라서 짧은 한 줄짜리 코드가
펜스에 둘러싸여 있을 때 유효한 2줄 이상 윈도우를 전혀 찾지 못하고 아무 경고 없이 완성문의
**첫 줄**(보통 산문)을 그대로 반환해버리는 실제 사례를 발견함(`print('hi')` 한 줄짜리
LCB 스타일 완성문 테스트에서 재현). 펜스를 먼저 벗겨내면 이 문제가 사라짐.

**검증:** 이미 수집해둔 실제 HumanEval 5문항 응답(재생성 없이, 저장된 raw 데이터 재사용)에
파이프라인의 실제 `_assemble_candidate_code`+`partial_pass_rate` 경로를 그대로 통과시킨
결과 — HumanEval/53·23·45·34: 0.00 → **1.00**, HumanEval/55(피보나치): 0.00 → 0.02
(모델 자체의 로직 오류, 추출 실패 아님). `tests/test_real_run.py`에 chat-style 완성문
케이스 2건 추가(HumanEval류·LCB류 각 1건), 기존 raw-continuation 케이스도 실제 evalplus
프롬프트 형태(닫힌 docstring 스텁)에 맞게 수정. 전체 pytest 126개 통과.

## 6. 문서 갱신

`pipeline/README.md`("What's built vs. what's deferred", 스모크 테스트 체크리스트,
Environments 섹션, §5 수정 완료로 갱신), `pipeline_build_plan.md`(BUILD STATUS 배너에
2026-08-15 항목 추가, §5 수정 완료로 갱신), `pipeline/requirements-h100.txt`(placeholder →
이번 실행의 `pip freeze` 기준 고정, gptqmodel/llmcompressor는 여전히 주석 처리 —
미설치·미검증) 전부 갱신.

## 7. GPTQ_AWQ_INT4 양자화 단계 구현 및 검증 — AWQ만, GPTQ는 구현 안 함 (같은 날 이어서)

**대상:** `models/loader.py`의 마지막 남은 `NotImplementedError` 스텁 `_load_gptq_or_awq`.
**정확히 말하면 "GPTQ/AWQ 백엔드"가 아니라 AWQ 백엔드다** — `Quant.GPTQ_AWQ_INT4`는 논문이
넷째 양자화 단계에 붙인 이름을 그대로 가져온 것일 뿐이고, 실제로 구현·검증한 건 AWQ(via
llm-compressor) 하나뿐이다. GPTQ(GPTQModel)는 §7.1에 적힌 이유로 시도 자체를 안 했다 —
"안 됨을 확인"이 아니라 "시도 안 함"이니 혼동하지 말 것. 상세 근거와 "나중에 시도해볼 것"은
`pipeline_build_plan.md`의 "Open assumptions" #1에 문서로 옮겨 기록(원래 `models/loader.py`
모듈 docstring에 길게 적어뒀던 걸, 코드가 아니라 문서에 있어야 한다는 지적을 받고 이동함).

사용자와 사전 합의한 범위: 메커니즘은 범용으로 구현하되 실제 검증은 Qwen2.5-7B와
Olmo3-7B(둘 다 7B급) 두 모델까지만 — Llama-3.1-8B와 32B 두 모델은 실제 파일럿 단계로 미룸.

### 7.1 open assumption #1 재해결 — 원래 계획과 다르게

`pipeline_build_plan.md`는 원래 GPTQModel(GPTQ)+llm-compressor(AWQ) 이원 구성에 모델별로
어느 쪽을 쓸지 나중에 정하기로 했었다. 실제 조사 결과 다르게 풀림:

- **GPTQModel 자체 소스(`gptqmodel/models/auto.py`)를 직접 fetch해서 확인 — `olmo3` 항목이
  아예 없다.** `olmo2`는 `LlamaQModel`(llama 클론)로 매핑되지만 `olmo`/`olmo3`는 없음. Olmo3
  arm에서 실패할 가능성이 높다고 판단.
- **llm-compressor는 아키텍처 전용 레지스트리가 없이** `AWQModifier`/`QuantizationModifier`를
  아무 HF `nn.Linear` 레이어에나 이름 패턴으로 적용하는 방식이라 Olmo3에서도 될 가능성이 높음.
- 논문 §4.3은 "GPTQ-int4 or AWQ-int4"를 사실상 동등한 넷째 조건으로 다루지 모델별로 다른
  기법을 요구하지 않음 → **다섯 모델 모두 llm-compressor(AWQ) 하나로 통일**하는 게 오히려
  더 일관된 설계라고 판단, 사용자 확인 받음.
- **캘리브레이션 데이터 도메인(코드 vs 일반 채팅)도 이론만으로 정하지 말고 실측 비교하기로**
  사용자가 제안 — Qwen2.5-7B를 두 도메인으로 각각 양자화해서 직접 비교.

### 7.2 GPTQ/AWQ의 구조적 차이 — bnb와 다른 2단계 구조

bnb는 로드할 때마다 즉시 양자화하지만, GPTQ/AWQ(llm-compressor)는 **캘리브레이션 데이터로
한 번 오프라인 양자화해서 디스크에 저장해두고, 그 이후엔 평범한
`AutoModelForCausalLM.from_pretrained(경로)`로 불러오는 방식**이다. 이 레시피는
`vllm-project/llm-compressor`의 공식 예제(`examples/awq/llama_example.py`)를 직접 fetch해서
확인(추측 아님): `AWQModifier(duo_scaling="both")` + `QuantizationModifier(scheme="W4A16_ASYM",
targets=["Linear"], ignore=["lm_head"])` 레시피를 `oneshot()`에 넘기고,
`model.save_pretrained(dir, save_compressed=True)`로 저장하면 이후 재로드는 특별한 로더
클래스 없이 그대로 `AutoModelForCausalLM.from_pretrained`로 된다.

이 구조 때문에 `scripts/quantize_model.py`(신규, 1회성 오프라인 스크립트)와
`_load_gptq_or_awq`(런타임 로더, 이미 양자화된 체크포인트가 없으면 `FileNotFoundError`로
명확히 안내)를 분리해서 구현. 양자화 자체는 절대 실제 실행 도중 암묵적으로 일어나지 않게
설계.

### 7.3 캘리브레이션 데이터셋 — 두 번 막히고 세 번째로 확정

원래 권장했던 `bigcode/the-stack-smol`(코드 도메인)은 **게이트 데이터셋**이라 인증 없이
막힘(`DatasetNotFoundError`, 실제로 돌려보고 발견). 대안으로 시도한
`codeparrot/github-code-clean`은 **레거시 로딩 스크립트 방식이라 `datasets>=5`에서 아예 로드가
안 됨**(LiveCodeBench 로더가 이미 겪었던 것과 동일한 `RuntimeError: Dataset scripts are no
longer supported`). 여러 후보를 직접 로드 테스트해서 `flytech/python-codes-25k`로 최종
확정 — 게이트 없고, `text` 필드에 "지시문 + 짧은 설명 + ```python 코드펜스" 형태가 이미
들어있어서 오히려 우리 파이프라인의 실제 추론 시점 분포(-Instruct 모델이 코드 생성
지시에 답하는 형태)에 더 가깝다는 걸 확인.

### 7.4 실행 중 만난 환경 문제 2건

1. **Triton CUDA 커널 컴파일이 `Python.h` 없어서 실패.** 이 머신의 `python3`가
   deadsnakes PPA로 설치된 3.12.13인데 `python3.12-dev`가 안 깔려 있었음(`apt-get install -y
   python3.12-dev`로 해결. 참고로 처음에 잘못 짚어서 `python3-dev`를 먼저 깔았는데, 이건
   우분투 기본 python3.10용 헤더라 실제로는 도움이 안 됐음).
2. **`pip install llmcompressor`가 torch/transformers/numpy를 조용히 업그레이드함**
   (torch 2.6.0+cu124 → 2.13.0+cu130, transformers 5.15.0 → 5.14.1, numpy 2.5.2 → 2.4.6).
   `requirements-h100.txt` 1차 고정(§6) 이후에 일어난 일이라, 이후 모든 GPTQ/AWQ 작업은
   새 버전으로 돌아갔다. bnb 스모크 테스트를 새 버전에서 재확인(peak 6.57GB, 체크리스트 전부
   통과) — 회귀 없음을 확인한 뒤 `requirements-h100.txt`를 새 버전 기준으로 다시 고정.

### 7.5 실제 실행 결과

세 번 양자화 실행(전부 성공, Olmo3도 별도 조치 없이 그대로 동작):

| 모델 | 캘리브레이션 | 체크포인트 크기 | 샘플 생성 |
|---|---|---|---|
| Qwen2.5-7B-Instruct | code | 5.2GB | 정상(사칙연산 함수) |
| Qwen2.5-7B-Instruct | chat | 5.2GB | 정상 |
| Olmo3-7B-Instruct | code | 4.7GB | 정상 |

`scripts/run_smoke_test.py`에 `--quant`/`--checkpoint-path` 플래그를 추가해 동일 체크리스트를
AWQ 체크포인트에도 그대로 재사용(3개 전부 체크리스트 8항목 통과, 문항별 pass_rate/탐지기
점수도 콘솔에 출력하도록 개선).

**캘리브레이션 도메인 비교 결과 (Qwen2.5-7B, 5문항):**

| 캘리브레이션 | pass_rate (5문항) | peak 메모리 |
|---|---|---|
| code | 1.00, 0.00, 1.00, 1.00, 1.00 | 16.12GB |
| chat | 1.00, 0.02, 1.00, 1.00, 1.00 | 16.02GB |

n=5로는 통계적으로 아무것도 판별할 수 없는 크기고(실제로 둘 다 같은 문항에서 실패), **이
표본에서는 코드 도메인 캘리브레이션이 일반 채팅 대비 감지 가능한 우위를 보이지 않았다.**
원 가설(코드 도메인이 유리할 것)을 반증하지도 않으므로, code-calibration 쪽을 canonical
경로(`data/quantized/<model>-awq/`)로 채택 — 실제 파일럿 스케일에서 재비교가 필요한 열린
질문으로 남김.

### 7.6 실행 중 발견해 즉시 고친 버그 2건 (스모크 테스트 스크립트 자체)

계획에는 없었지만 두 체크포인트를 나란히 비교하는 과정에서 실제로 만난 것들:

| 문제 | 원인 | 수정 |
|---|---|---|
| 두 번째 비교 실행이 `score_logprobs() called before generate()`로 죽음 | `GenerationCache`의 캐시 키가 `--checkpoint-path`를 반영 안 해서, 서로 다른 체크포인트인데도 같은 `(model, quant)`로 캐시 히트 — 두 번째 실행의 모델 인스턴스는 `generate()`가 실제로 호출된 적이 없는데 `score_logprobs()`만 불림 | `quant_label`을 도입해 체크포인트 경로가 다르면 캐시 키·출력 디렉터리도 달라지게 함. 근본적으로는 재실행마다 임시 캐시 디렉터리를 새로 만들도록 변경(이 스크립트는 재현성보다 "매번 실제 경로를 탄다"가 목적이므로) |
| 세 번째 재실행도 같은 에러로 죽음 | 위 수정 후에도 **같은 명령을 두 번 돌리면** 두 번째 실행이 첫 번째 실행이 남긴 캐시를 정당하게 히트하면서 동일한 근본 문제가 재현됨 — `score_logprobs()`가 "이 프로세스에서 방금 generate()를 호출했는가"를 메모리로만 추적하는 설계가 프로세스 간 캐시 재사용과 근본적으로 안 맞음 | 캐시를 프로세스 간 공유하지 않도록 매 실행마다 `tempfile.mkdtemp()`로 새 캐시 디렉터리 사용. 실제 파이프라인(`real_run.py`/`dry_run.py`)은 현재 `score_logprobs()`를 아예 안 부르므로 이 문제가 없지만, 나중에 쓰게 되면 같은 함정에 빠질 수 있음 — README에 명시 |

또한 AWQ 체크포인트의 peak GPU 메모리가 16GB 안팎으로 nf4(2-12GB 기대)와 확연히 다르게
나와서 체크리스트가 계속 실패했는데, 원인을 llm-compressor GitHub 이슈 #1550(비대칭
zero-point 압축 해제 관련 알려진 한계)로 특정하고, `PLAUSIBLE_PEAK_GB`를 quant별 딕셔너리로
바꿔 정직하게 반영(임의로 통과시키지 않음).

## 8. 남은 것

- Llama-3.1-8B-Instruct, Qwen2.5-32B, Olmo3-32B — GPTQ/AWQ 미양자화(의도적으로 이번 범위 밖)
- §5의 LCB 마크다운 추출 경로 — 실측 LCB 응답으로 아직 검증 안 됨(HumanEval만 실측)
- 캘리브레이션 도메인(코드 vs 채팅) 비교 — n=5라 결론 낼 수 없음, 실제 파일럿 스케일에서
  재확인 필요
- 실제 파일럿 스케일 실행(§7 6단계) — 이번 세션은 5문항 스모크 테스트까지만
