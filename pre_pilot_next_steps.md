# Pre-Execution Validation and Main-Run Steps

> 작성일: 2026-08-19
> 상태: 실행 전(pre-execution) 로드맵
> 목적: H100 본 실행 전에 구현 검증과 연구 데이터 생성을 분리하고, 프로토콜·데이터·분석 작업의 순서를 고정한다.
> 주의: 이 문서의 수치는 별도 표기가 없으면 계획값 또는 데이터 가용성 점검값이다. 실험 결과가 아니다.

## 상태

고정 프롬프트 채점, BF16 기준선, LiveCodeBench 출력 경로, 모델–문항 시간 라벨,
코퍼스 검색 3상태 스키마는 구현됐다. 실제 연구 원자료는 아직 없다. 기존 파일럿 집계 경로는
구현 검증과 연구 추정을 섞을 위험이 있어 폐기했으며, 다음 작업은 Olmo 코퍼스 검색·TRACER
충실도 확인, 제한된 구현 스모크 테스트, 운영 구성 고정, 본 실행이다.

## 1. 본 실행 전 작업

### 1.1 확률 기반 탐지기의 채점 대상을 고정 프롬프트로 변경

arXiv:2603.03203은 perplexity와 Min-k% Prob를 각 **test prompt**에 대해 계산한다. 정본도
두 탐지기를 고정 텍스트에 대한 teacher-forced 로그확률로 정의한다
([paper/paper_draft.md](paper/paper_draft.md), §4.4).

2026-08-20 수정 전 [pipeline/src/qcd/real_run.py](pipeline/src/qcd/real_run.py)는 다음과 같이 동작했다.

- greedy 생성문의 `token_logprobs`로 perplexity와 Min-k% Prob를 계산한다.
- 구현된 `score_logprobs()`는 당시 실제 실행 루프에서 호출되지 않았다.
- 따라서 탐지기 값이 모델이 생성한 서로 다른 텍스트에 의존하며, 정밀도 간 고정 텍스트
  대응 비교가 깨진다.

사전 고정할 프로토콜:

1. 채점 텍스트는 벤치마크 프롬프트로 한다.
2. 모든 정밀도에서 동일한 텍스트를 사용한다.
3. instruct 모델의 chat wrapper는 문맥으로만 제공한다.
4. perplexity와 Min-k% 통계에는 실제 프롬프트 토큰만 포함한다.
5. 생성 캐시와 독립적인 `score_prompt_logprobs(prompt)` 계열 API를 둔다.
6. 실제 실행 루프가 이 API를 호출하는 회귀 테스트를 추가한다.

**상태: 2026-08-20 완료.** `score_prompt_logprobs(item_id, prompt)`를 생성
이력과 독립된 API로 구현했고, instruct chat wrapper는 문맥으로 유지하되 offset
mask로 실제 문제 토큰만 반환한다. `perplexity`와 `mink_prob`은 이 고정 프롬프트
배열을 사용한다. 생성 답변 로그확률은 `generations.parquet`의 `token_logprobs`와
탐색적 `completion_perplexity`/`completion_mink_prob`으로 별도 보존하며, 고정
프롬프트 원배열은 greedy 행의 `prompt_token_logprobs`에 저장한다. 전체 테스트
133개와 Qwen2.5-7B BNB-nf4 실제 H100 5문항 smoke test가 통과했다.

### 1.2 16-bit 기준선을 bf16으로 확정

2026-08-20 점검에서 Qwen2.5-7B/32B와 Olmo3-7B의 배포 dtype이 bf16임을
확인했고, H100의 자연스러운 실행 경로와도 일치하므로 **bf16을 기준선으로 확정**했다.
`Quant.BF16="bf16"`으로 이름을 바꾸고 `dtype=torch.bfloat16`을 강제하여
manifest의 표기와 실제 로드 dtype이 일치하게 했다. 다음 항목을 함께 맞췄다.

- `paper/paper_draft.md`
- `paper/paper_draft_ko.md`
- `pipeline/src/qcd/config.py`
- 모델 로더와 manifest의 precision 표기
- 관련 테스트와 표·그림 캡션

영문·한국어 정본과 관련 표·테스트도 bf16으로 동기화했다. Olmo3.1-32B repo ID는
`allenai/Olmo-3.1-32B-Instruct`로 확인해 registry에 반영했다. 7B(Olmo 3/1025)와
함께 기존 설계의 7B–32B 크기 축을 구성한다.
Llama-3.1 gated access는 별도 모델 가용성 문제로 남아 있으며 dtype 결정을 바꾸지는 않는다.

### 1.3 개발 전용 집계 경로 격리

**상태: 2026-08-28 수정 완료.** 기존 [pipeline/scripts/run_pilot.py](pipeline/scripts/run_pilot.py)와
`pipeline/scripts/aggregate_pilot.py`는 구현 검증 출력에서 효과 크기·AUC·기저율·상관·검정력을
계산할 수 있어 연구 파일럿처럼 오해될 위험이 있었다. 두 스크립트는 이제 실행을 거부하는 폐기
호환 진입점이다. 로컬 드라이런은 합성 진단만, 실제 하드웨어 스모크 테스트는 로딩·호환성·스키마·
유한 로그확률·샌드박스·메모리·처리량만 확인한다.

개발 내부 집계 보조 코드는 회귀 테스트용으로만 남아 있으며 산출물 이름과 상태를
`development_summary.json`, `development_power_diagnostics.json`,
`development_only_not_manuscript_evidence`로 바꿨다. 이 경로는 본 실행의 표본 수, CDD 자격,
C1–C4 또는 논문 결과를 바꾸지 않는다.

Olmo3 코퍼스 검색 결과는 연구 효과 추정과 분리하여 `confirmed-match`,
`no-match-found`, `not-observable` 상태로 기록한다. `no-match-found`를 비노출로 간주하지
않으므로 오류율 *e*, 위양성률, 위음성률을 계산하거나 임시 값으로 채우지 않는다.

### 1.4 LiveCodeBench 실제 출력 경로 smoke test

**상태: 2026-08-20 완료.** Qwen2.5-7B BNB-NF4로 pre/post 각 1개씩, stdin/functional
각 2개씩 총 4문항을 실제 H100에서 실행했다.

본 실행 전에 Qwen2.5-7B BNB-nf4로 각 유형 1–5문항을 구현 검증용으로 실행하여 다음만 확인한다.

- markdown fence 및 산문 제거
- stdin 프로그램 보존
- `Solution` 메서드 추출
- 공개·비공개 테스트 디코딩
- 부분 통과율과 pass@1 기록

첫 실행에서 LCB 생성 프롬프트가 `starter_code`와 실행 형식 지시를 누락한 문제를 발견했다.
고정 문제 확률 채점은 원문 `item.prompt`를 유지하고, 생성에만 functional starter code 또는
stdin 입출력 계약을 추가하도록 분리했다. 수정 후 네 문항 모두 구조 검사와 Parquet 왕복을
통과했다. 이때 생성된 pass@1과 부분 점수는 구현 검증 부산물이며 논문 결과나 설계 입력으로
사용하지 않는다.

### 1.5 문서와 코드의 낡은 상태 설명 정리

GPU 경로가 이미 구현됐는데도 일부 docstring과 CLI 설명에는 `NotImplementedError`가 남아
있다. 이는 실행 판단을 혼란스럽게 하므로 본 실행 전 구현 감사에서 함께 정리한다. 이 작업은
측정 로직 수정 후 수행하며, 구현 상태의 정본은 [pipeline/README.md](pipeline/README.md)로
통일한다.

## 2. LiveCodeBench 표본 수에 따른 설계 결정

### 2.1 확인된 데이터 가용성

공식 `code_generation_lite`의 현재 최신 배포본은 총 **1,055문항**이다. 따라서 논문의
“pre/post 각각 ≥1,000” 목표는 이 배포본으로는 산술적으로 불가능하다. 공식 v7도 아직
배포되지 않았고 공개 요청 이슈가 열린 상태다.

- 공식 데이터셋: <https://huggingface.co/datasets/livecodebench/code_generation_lite/tree/main>
- v7 요청: <https://github.com/LiveCodeBench/LiveCodeBench/issues/139>

2026-08-20에 현재 로더로 직접 센 값:

| 경계 | pre | post | 총합 |
|---|---:|---:|---:|
| 2023-03-01 | 0 | 1,055 | 1,055 |
| 2023-12-01 | 286 | 769 | 1,055 |
| 2024-09-20 | 690 | 365 | 1,055 |
| 2025-01-01 | 873 | 182 | 1,055 |

동일 `release_v6`를 모델–문항 주 경계로 다시 분류한 가용성 수:

| 모델 계열 | `possible-exposure` | `clean-by-model-cutoff` | `shared-clean-control` |
|---|---:|---:|---:|
| Qwen2.5 | 690 | 183 | 182 |
| Llama-3.1 주 경계 | 326 | 547 | 182 |
| Olmo3 | 873 | 0 | 182 |

Llama 민감도 경계에서는 LCB `possible-exposure`가 0개이므로 해당 민감도 Q1b/Q2는 추정 불가다.

Olmo3-7B-Instruct와 Olmo3.1-32B-Instruct의 공식 모델 카드는 모두 `Date cutoff: Dec. 2024`를
명시한다. 두 카드가 Base→SFT→DPO→RLVR의 최종 Instruct 계보에 붙은 모델 수준 cutoff이므로
공통 경계에 적용한다. 월 단위 선언에서는 12월 전체를 노출 가능 구간으로 보고
**2025-01-01**을 첫 post-cutoff 날짜로 사용한다.

Olmo 3 기술 보고서도 pretraining Common Crawl이 `CC-MAIN-2024-51`에서 끝나고, PDF 수집
cutoff가 2024년 12월이라고 명시한다. post-training은 Dolci SFT·DPO·RLVR로 구성되며 모델
카드의 Dec. 2024 cutoff가 최종 체크포인트 수준의 경계를 제공한다. 코퍼스 참조 검색에서는
각 단계를 별도로 검색하고, 확인된 일치가 시간 대리 라벨과 정합하는지 기술적으로 보고한다.

- Ai2 Dolma 3 문서: <https://docs.allenai.org/in_depth/pretraining>
- Olmo3-7B-Instruct 모델 카드: <https://huggingface.co/allenai/Olmo-3-7B-Instruct>
- Olmo3.1-32B-Instruct 모델 카드: <https://huggingface.co/allenai/Olmo-3.1-32B-Instruct>
- Olmo 3 기술 보고서: <https://arxiv.org/abs/2512.13961>

Qwen2.5는 명확한 공식 cutoff 선언이 없으므로 공식 출시일 **2024-09-19**를 보수적 경계로
사용한다. 일 단위 라벨에서는 출시 당일을 제외하여 LCB-post를 **2024-09-20**부터 시작한다.

- Qwen2.5 공식 발표: <https://qwenlm.github.io/blog/qwen2.5/>

Qwen2.5보다 Olmo3 경계가 늦으므로 공통 경계는 **2025-01-01**이다.
본 실행 구성의 공통 경계도 이 날짜로 고정했다.

### 2.2 지금 고정할 분석 지위

#### Q1a

주 분석으로 유지한다. 오염 라벨이 필요 없고, 1,055문항 전체에 대해 같은 문항의 정밀도별
점수 이동을 측정할 수 있다. 현재 데이터 가용성에서 가장 확실하게 보고 가능한 결과다.

#### Q1b

`possible-exposure`와 `shared-clean-control`을 구분하는 **대리 라벨 AUC** 분석으로 유지한다.
post=182라면 대리 라벨 기준 ΔAUC=0.05의 필요량 170을 겨우 넘으므로 효과 크기가 작을 때
검정력이 제한될 수 있다. 다음 중 하나가 필요하다.

- 실제 효과가 0.05보다 큼
- 더 많은 동일 계열 데이터 확보
- 검정력 부족을 인정하고 신뢰구간·기술 결과로 보고

*e*=0–30% 표는 실제 오염 AUC로 재해석하려 할 때의 구성 타당도 민감도 분석으로만 사용한다.
Olmo 코퍼스 비검출로 *e*를 식별할 수 없으므로 어떤 arm도 표의 특정 행에 경험적으로
배정하지 않으며, 이 표를 데이터 의존적 사이징 입력으로 사용하지 않는다.

#### Q2

정본 §5의 사전 규칙대로 **신뢰구간 중심 보조 분석**으로 강등한다. 새 데이터가 확보되지
않는 한 유의성에 논문의 기여 주장을 걸지 않는다.

다른 데이터셋을 섞어 표본 수를 늘리면 “같은 출처와 형식에서 날짜만 다르다”는 핵심 통제가
깨진다. 이 선택은 단순한 데이터 추가가 아니라 중대한 사전 지정 설계 변경으로 다뤄야 한다.

## 3. Olmo3 경계·코퍼스 참조·TRACER

프로토콜 수정 후 다음 순서로 진행한다.

1. Olmo3 공식 모델 카드의 Dec. 2024 cutoff를 확인했다. **완료.**
2. 월 단위 보수 경계 2025-01-01로 LCB pre/post를 다시 계산하고 경계를 동결했다. **완료.**
3. Qwen2.5의 주 경계는 출시일 2024-09-19로 동결하고, 운영상 LCB-post 시작일은
   2024-09-20으로 사용한다.
4. Olmo3 코퍼스 참조 검색과 문항별 검색 상태 기록을 구축한다.

Olmo3 코퍼스 참조는 다음 세 계열을 별도로 실행하고 결과를 합성 평균하지 않는다.

- exact 및 *n*-gram 일치
- edit-distance 및 AST 기반 surface/semantic 일치
- embedding retrieval 후 LLM paraphrase 판정

pretraining과 post-training을 분리 보고한다. Olmo3 Instruct 모델은 어느 단계에서도
벤치마크 정보를 흡수할 수 있기 때문이다.

Ai2의 OLMoTrace는 Olmo3 출력과 전체 훈련 데이터의 verbatim span을 연결할 수 있으므로,
exact-match 후보 검색 백엔드로 사용할 수 있는지 먼저 조사한다. 다만 OLMoTrace만으로
AST 또는 의미적 중복을 대체하지 않는다.

- Olmo3/OLMoTrace 설명: <https://allenai.org/blog/olmo3>

그 뒤 TRACER를 원문 명세대로 재구현하고 다음을 공개 코퍼스의 확인된 양성 근거와 비교한다.

- `confirmed-match` 문항에서 시간 대리 라벨과 TRACER 결과의 정합성
- TRACER 재구현과 공개 코퍼스 검색의 양성 탐지 일치·불일치
- exact와 semantic/paraphrase 탐지 간 불일치

이 비교는 Q1b 대리 라벨의 구성 타당도를 보조하는 기술적 증거다. 검증된 음성 라벨이 없으므로
실제 오염 AUC, 오류율 *e*, 위양성률, 위음성률을 산출하지 않는다. Q1a와 Q1b의 계산 자체는
이 단계와 독립적이지만, Q1b를 실제 오염 판별 성능으로 확대 해석하지 않는다.

## 4. 권장 실행 순서

1. 확률 탐지기를 고정 프롬프트 teacher-forcing으로 수정한다. **완료.**
2. bf16 기준선을 확정하고 영문·한국어 정본과 코드를 동기화한다. **완료.**
3. 구현 검증과 연구 추정을 섞던 파일럿 집계 진입점을 폐기하고 개발 전용 진단으로 격리한다. **완료.**
4. LCB stdin/functional 실응답 smoke test를 수행한다. **완료.**
5. LCB 표본 수와 Q2의 분석 지위를 영문·한국어 정본에 동시 반영한다. **완료.**
6. Olmo3 날짜 경계, 문자열 검색 1차 구현, 문항별 3상태 기록은 완료했다. 전수 코퍼스 검색과
   표면·의미·패러프레이즈 계열은 아직 실행·구현 전이다.
7. TRACER 재구현 결과를 공개 코퍼스의 확인된 양성 근거와 비교한다.
8. Qwen2.5-7B와 Olmo3-7B의 bf16·BNB-nf4를 제한된 H100 스모크 테스트로 실행하되, 결과값이
   아니라 로딩·형식·메모리·처리량 실패만 점검한다.
9. 결과 확인 전에 필요한 운영 변경을 기록하고 모델·문항·채점·분석 구성과 C1–C4를 고정한다.
10. 다섯 모델 × 네 정밀도의 본 실행을 수행한다. 이 실행만 연구 데이터의 출처다.
11. 본 실행 완료 후에만 사전 지정 분석과 달성 정밀도 평가를 수행하며 사후 사이징은 하지 않는다.

## 5. 단계별 완료 조건

### 구현 검증 감사 완료

- probability detector가 생성문이 아닌 고정 프롬프트를 채점한다.
- manifest의 16-bit dtype과 실제 모델 dtype이 일치한다.
- LCB 두 실행 유형의 실제 출력 smoke test가 통과한다.
- 검증 출력이 연구 효과 추정·CDD 관문·검정력 재계산으로 연결되지 않는다.
- 전체 테스트가 통과한다.

### 라벨링 단계 완료

- Olmo3 최종 날짜 경계가 출처와 함께 기록된다.
- pretraining과 post-training 코퍼스 검색 결과가 분리 저장된다.
- 세 탐지 계열의 결과와 불일치가 문항 단위로 저장된다.
- 각 문항의 `confirmed-match`, `no-match-found`, `not-observable` 상태와 TRACER 비교 결과가
  저장되며, 비검출에서 오류율 *e*를 계산하지 않는다.

### 본 실행 전 검증 완료

- 로컬 드라이런과 제한된 H100 스모크 테스트가 구현 검증 항목을 통과한다.
- 검증 출력은 별도 경로에 저장되고 논문 결과·효과 크기·AUC·pass rate·검정력에 사용되지 않는다.
- CDD와 확률 채점의 처리량은 운영 계획용으로만 분리 기록한다.
- 결과 확인 전에 운영 설정, 문항 집합, CDD 샘플 수와 예상 wall-clock을 동결한다.
- C1–C4의 자격과 표본 계획이 검증 출력으로 변경되지 않는다.
