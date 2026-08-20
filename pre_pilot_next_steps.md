# Pre-Pilot Next Steps

> 작성일: 2026-08-19
> 상태: 실행 전(pre-execution) 로드맵
> 목적: 실제 H100 파일럿 전에 해결해야 할 프로토콜·데이터·분석 작업을 우선순위대로 고정한다.
> 주의: 이 문서의 수치는 별도 표기가 없으면 계획값 또는 데이터 가용성 점검값이다. 실험 결과가 아니다.

## 상태

고정 프롬프트 채점과 BF16 기준선 전환은 완료됐다. 실제 실험 원자료는 아직 없다. 다음 작업은
파일럿 집계기 구현과 LiveCodeBench 출력 경로 검증이다.

## 1. 파일럿 전 작업

### 1.1 확률 기반 탐지기의 채점 대상을 고정 프롬프트로 변경

arXiv:2603.03203은 perplexity와 Min-k% Prob를 각 **test prompt**에 대해 계산한다. 정본도
두 탐지기를 고정 텍스트에 대한 teacher-forced 로그확률로 정의한다
([paper/paper_draft.md](paper/paper_draft.md), §4.4).

2026-08-20 수정 전 [pipeline/src/qcd/real_run.py](pipeline/src/qcd/real_run.py)는 다음과 같이 동작했다.

- greedy 생성문의 `token_logprobs`로 perplexity와 Min-k% Prob를 계산한다.
- 구현된 `score_logprobs()`는 실제 파일럿 루프에서 호출되지 않는다.
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

### 1.3 실제 파일럿 집계기 완성

**상태: 2026-08-20 구현 완료.** [pipeline/scripts/run_pilot.py](pipeline/scripts/run_pilot.py)는
원자료 생성 후 parquet 집계기를 호출한다. 저장된 원자료만 다시 집계할 때는
`pipeline/scripts/aggregate_pilot.py`를 사용한다.

- (a) Q1a 탐지기별 대응 Cohen's *d*
- (b) Q1b의 16-bit AUC와 정밀도 간 상관 *r*
- (c) Q2 로그오즈 교차효과와 문항 수준 *r*
- (d) 조건별·모델별 실제 pass@1 기저율
- (e) Olmo3 대리 라벨 오류율 *e*
- CDD 관문 판정(AUC ≥ 0.7936)
- 정밀도별 처리량: CDD 다중 생성과 확률 채점 분리
- 전체 실행 wall-clock 추정
- 파일럿 값에 따른 검정력 재계산

저장 산출물:

- `pilot_summary.json`
- `power_recompute.json`
- 명시적 pass@1 필드
- 모델·토크나이저 revision과 데이터 snapshot

Olmo3 오류율 *e*는 corpus ground truth 구축 전까지 `null`과
`pending_corpus_ground_truth`로 기록한다. 임시 값을 넣지 않는다.

### 1.4 LiveCodeBench 실제 출력 경로 smoke test

HumanEval의 markdown 코드 추출과 샌드박스 실행은 실제 H100 출력으로 검증되었다. 반면
LiveCodeBench의 stdin형과 functional형은 실제 모델 출력으로 아직 검증되지 않았다.

파일럿 전에 Qwen2.5-7B BNB-nf4로 각 유형 1–5문항을 실행하여 다음을 확인한다.

- markdown fence 및 산문 제거
- stdin 프로그램 보존
- `Solution` 메서드 추출
- 공개·비공개 테스트 디코딩
- 부분 통과율과 pass@1 기록

### 1.5 문서와 코드의 낡은 상태 설명 정리

GPU 경로가 이미 구현됐는데도 일부 docstring과 CLI 설명에는 `NotImplementedError`가 남아
있다. 이는 실행 판단을 혼란스럽게 하므로 pre-pilot audit에서 함께 정리한다. 이 작업은
측정 로직 수정 후 수행하며, 구현 상태의 정본은 [pipeline/README.md](pipeline/README.md)로
통일한다.

## 2. LiveCodeBench 표본 수에 따른 설계 결정

### 2.1 확인된 데이터 가용성

공식 `code_generation_lite`의 현재 최신 배포본은 총 **1,055문항**이다. 따라서 논문의
“pre/post 각각 ≥1,000” 목표는 이 배포본으로는 산술적으로 불가능하다. 공식 v7도 아직
배포되지 않았고 공개 요청 이슈가 열린 상태다.

- 공식 데이터셋: <https://huggingface.co/datasets/livecodebench/code_generation_lite/tree/main>
- v7 요청: <https://github.com/LiveCodeBench/LiveCodeBench/issues/139>

2026-08-19에 현재 로더로 직접 센 값:

| 경계 | pre | post | 총합 |
|---|---:|---:|---:|
| 2023-03-01 | 0 | 1,055 | 1,055 |
| 2023-12-01 | 286 | 769 | 1,055 |
| 2024-09-20 | 690 | 365 | 1,055 |
| 2024-12-31 | 873 | 182 | 1,055 |

Dolma 3의 Common Crawl 원천은 `CC-MAIN-2024-51`까지 포함한다. 따라서 Olmo3의 최종
보수적 경계가 2024년 말 부근이면 post 조건은 약 182문항까지 줄어들 수 있다. 이는
Common Crawl 부분만을 이용한 **잠정 추론**이며, 실제 경계는 pretraining의 다른 소스와
midtraining·post-training 데이터의 날짜까지 확인해 확정해야 한다.

- Ai2 Dolma 3 문서: <https://docs.allenai.org/in_depth/pretraining>

Qwen2.5는 LLMLagBench 관리자 평가를 요청했으나 2026-08-19까지 결과가 없었다. 사전 지정한
fallback에 따라 공식 출시일 **2024-09-19**를 Qwen2.5의 주 경계로 동결한다. 일 단위 라벨에서는
출시 당일을 제외하여 LCB-post를 **2024-09-20**부터 시작한다. 이후 관리자 추정치가 도착하면
주 경계를 바꾸지 않고 민감도 분석에만 사용한다.

- Qwen2.5 공식 발표: <https://qwenlm.github.io/blog/qwen2.5/>

구현 후속 작업: `pipeline/scripts/run_pilot.py`의 현재 임시 기본값 `2024-09-01`은 아직 코드에서
바뀌지 않았다. 파일럿 실행 전 이를 `2024-09-20`으로 수정하거나 CLI에
`--lcb-cutoff 2024-09-20`을 명시해야 한다. 전체 주 분석에는 Olmo3 경계까지 확정한 뒤 두
상한 중 더 늦은 날짜를 전달한다.

### 2.2 지금 고정할 분석 지위

#### Q1a

주 분석으로 유지한다. 오염 라벨이 필요 없고, 1,055문항 전체에 대해 같은 문항의 정밀도별
점수 이동을 측정할 수 있다. 현재 데이터 가용성에서 가장 확실하게 보고 가능한 결과다.

#### Q1b

라벨 품질과 효과 크기에 조건부인 분석으로 유지한다. post=182라면 무잡음 기준
ΔAUC=0.05의 필요량 170을 겨우 넘는다. 그러나 *e*=10%이면 필요량은 287이므로 동일한
효과를 검출하기 어렵다. 다음 중 하나가 필요하다.

- 실제 효과가 0.05보다 큼
- 측정된 *e*가 거의 0에 가까움
- 더 많은 동일 계열 데이터 확보
- 검정력 부족을 인정하고 신뢰구간·기술 결과로 보고

#### Q2

정본 §5의 사전 규칙대로 **신뢰구간 중심 보조 분석**으로 강등한다. 새 데이터가 확보되지
않는 한 유의성에 논문의 기여 주장을 걸지 않는다.

다른 데이터셋을 섞어 표본 수를 늘리면 “같은 출처와 형식에서 날짜만 다르다”는 핵심 통제가
깨진다. 이 선택은 단순한 데이터 추가가 아니라 사전 등록 수준의 설계 변경으로 다뤄야 한다.

## 3. Olmo3 경계·ground truth·TRACER

프로토콜 수정 후 다음 순서로 진행한다.

1. Olmo3의 pretraining, midtraining, long-context, SFT, DPO, RLVR 데이터 전체에서 가장 늦은
   적격 날짜를 확인한다.
2. 그 날짜로 LCB pre/post 수를 다시 계산하고 경계를 동결한다.
3. Qwen2.5는 관리자 무응답 fallback 적용을 완료했다. 주 경계는 출시일 2024-09-19로 동결하고,
   운영상 LCB-post 시작일은 2024-09-20으로 사용한다.
4. Olmo3 corpus ground-truth 라벨링을 구축한다.

Olmo3 라벨링은 다음 세 계열을 별도로 실행하고 결과를 합성 평균하지 않는다.

- exact 및 *n*-gram 일치
- edit-distance 및 AST 기반 surface/semantic 일치
- embedding retrieval 후 LLM paraphrase 판정

pretraining과 post-training을 분리 보고한다. Olmo3 Instruct 모델은 어느 단계에서도
벤치마크 정보를 흡수할 수 있기 때문이다.

Ai2의 OLMoTrace는 Olmo3 출력과 전체 훈련 데이터의 verbatim span을 연결할 수 있으므로,
exact-match 후보 검색 백엔드로 사용할 수 있는지 먼저 조사한다. 다만 OLMoTrace만으로
AST 또는 의미적 중복을 대체하지 않는다.

- Olmo3/OLMoTrace 설명: <https://allenai.org/blog/olmo3>

그 뒤 TRACER를 원문 명세대로 재구현하고 다음을 ground truth에 대조한다.

- pre/post-cutoff proxy의 오류율 *e*
- TRACER 재구현의 탐지 성능
- exact와 semantic/paraphrase 탐지 간 불일치

Q1b는 이 단계가 완료되어야 확증적으로 해석할 수 있다. Q1a는 이 단계와 독립적이다.

## 4. 권장 실행 순서

1. 확률 탐지기를 고정 프롬프트 teacher-forcing으로 수정한다.
2. bf16 기준선을 확정하고 영문·한국어 정본과 코드를 동기화한다. **완료.**
3. 실제 파일럿 집계기와 power-recompute 산출물을 구현한다. **완료.**
4. LCB stdin/functional 실응답 smoke test를 수행한다.
5. LCB 표본 수와 Q2의 분석 지위를 영문·한국어 정본에 동시 반영한다.
6. Olmo3의 실제 날짜 경계와 corpus ground truth를 구축한다.
7. TRACER 재구현을 ground truth에 검증하고 *e*를 측정한다.
8. Qwen2.5-7B와 Olmo3-7B의 16-bit→BNB-nf4 파일럿을 실행한다.
9. CDD AUC ≥ 0.7936 관문을 판정한다.
10. 다섯 파일럿 값과 처리량으로 최종 표본 수와 CDD 샘플 수를 결정한다.
11. 그 뒤에만 다섯 모델 × 네 정밀도의 본 실험을 실행한다.

## 5. 단계별 완료 조건

### Pre-pilot audit 완료

- probability detector가 생성문이 아닌 고정 프롬프트를 채점한다.
- manifest의 16-bit dtype과 실제 모델 dtype이 일치한다.
- LCB 두 실행 유형의 실제 출력 smoke test가 통과한다.
- 실제 parquet에서 (a)–(d)를 계산하는 파일럿 리포터가 동작한다.
- 전체 테스트가 통과한다.

### 라벨링 단계 완료

- Olmo3 최종 날짜 경계가 출처와 함께 기록된다.
- pretraining과 post-training ground truth가 분리 저장된다.
- 세 탐지 계열의 결과와 불일치가 문항 단위로 저장된다.
- TRACER fidelity와 proxy-label 오류율 *e*가 계산된다.

### 파일럿 완료

- (a)–(e)가 모두 측정된다.
- CDD 관문 결과가 기록된다.
- CDD와 확률 채점의 처리량이 분리 기록된다.
- `pilot_summary.json`과 `power_recompute.json`이 생성된다.
- 본 실험의 문항 수, CDD 샘플 수, 예상 wall-clock이 동결된다.
