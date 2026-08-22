# 업데이트 내역 (2026-08-20)

> 이 문서는 2026-08-20 작업 종료 시점의 스냅샷이다. 이후 완료 상태는
> `pre_pilot_next_steps.md`와 `pipeline/README.md`를 따른다.

## 요약

1. 오염 탐지용 **고정 문제 프롬프트 확률**과 생성 답변의 **confidence**를 분리했다.
2. 이름과 실제 실행 dtype이 달랐던 16-bit 기준선을 **BF16**으로 확정했다.

논문의 Q1/Q2 구조, CDD 공식, pass@1 계산, 오염 라벨과 통계 설계는 변경하지 않았다.

## 1. 변경 전 문제

기존 파이프라인은 greedy 생성 답변의 `token_logprobs`로 Perplexity와 Min-k% Prob를
계산했다.

```text
문제 → greedy 답변 생성 → 생성 답변 로그확률 → Perplexity / Min-k
```

이 값은 모델이 자신이 생성한 답변을 얼마나 높은 확률로 냈는지 보여주는 confidence
신호로는 유용하다. 하지만 BF16과 NF4가 서로 다른 답변을 생성하면 텍스트, 길이, 형식,
토큰 난이도 차이가 모두 섞이므로, 같은 문제에 대한 오염 탐지 신호를 정밀도 간 직접
비교하기 어렵다.

정본 §4.4는 Perplexity와 Min-k% Prob를 모든 정밀도에서 동일한 고정 텍스트에 대한
teacher-forced 로그확률로 정의한다. 기존 구현은 이 프로토콜과 일치하지 않았다.

## 2. 변경 후 측정 구조

이제 각 문항에서 다음 신호를 분리해 측정한다.

```text
벤치마크 문제
├─ 고정 문제 토큰 로그확률
│  ├─ Perplexity          주 오염 탐지 점수
│  └─ Min-k% Prob         주 오염 탐지 점수
│
└─ 생성 답변
   ├─ 답변 토큰 로그확률
   │  ├─ Completion Perplexity   탐색적 confidence
   │  └─ Completion Min-k%       탐색적 confidence
   ├─ 반복 생성 유사도            CDD
   └─ 코드 테스트 결과            pass@1 / partial pass rate
```

주 Q1 분석에는 기존 세 탐지기를 사용한다.

- `cdd`
- `perplexity`
- `mink_prob`

생성 답변 confidence는 다음 이름으로 별도 저장한다.

- `completion_perplexity`
- `completion_mink_prob`

## 3. 고정 프롬프트 채점 API

`pipeline/src/qcd/models/loader.py`의 `LoadedModel` 프로토콜과 실제 모델 어댑터에
다음 API를 추가했다.

```python
score_prompt_logprobs(
    item_id: str,
    prompt: str,
) -> list[float]
```

기존 `score_logprobs(item_id, generated_token_ids)`는 생성 답변 confidence 채점용으로
유지한다. 새 API는 생성 이력과 독립적이므로 generation cache hit로 `generate()`가
생략된 경우에도 안전하게 사용할 수 있다.

## 4. Instruct chat wrapper 마스킹

Qwen, Llama, Olmo Instruct 모델에는 정상적인 chat template을 적용한다.

```text
<user>                  통계에서 제외
벤치마크 문제            통계에 포함
</user><assistant>       통계에서 제외
```

모델 forward pass에는 전체 wrapper를 문맥으로 제공하지만, tokenizer offset mapping으로
실제 벤치마크 문제 범위에 완전히 포함되는 토큰만 선택해 로그확률 배열을 만든다.
Offset mapping을 지원하지 않는 느린 tokenizer를 위한 prefix-span fallback도 추가했다.

## 5. 원자료 저장 변경

`generations.parquet`에는 두 종류의 로그확률을 구분해 저장한다.

| 컬럼 | 의미 |
|---|---|
| `token_logprobs` | 생성 답변 토큰의 로그확률 |
| `prompt_token_logprobs` | 고정 문제 토큰의 로그확률 |

`prompt_token_logprobs`는 동일 배열의 중복을 피하기 위해 문항별 greedy generation 행에만
저장하며 sampled generation 행에는 `null`을 기록한다.

`detector_scores.parquet`에는 문항별로 다음 다섯 점수가 저장된다.

```text
cdd
perplexity
mink_prob
completion_perplexity
completion_mink_prob
```

`perplexity` detector score는 AUC 방향을 맞추기 위해 실제로는
`-log(perplexity)`, 즉 평균 로그확률을 사용한다. 따라서 값이 높을수록 오염 의심
방향이다. 일반적인 사람이 읽는 perplexity 값과 방향이 반대라는 기존 규약은 유지했다.

## 6. 실제 실행 루프

`pipeline/src/qcd/real_run.py`는 이제 다음 순서로 처리한다.

1. greedy 답변 1개 생성
2. temperature 0.8 답변 여러 개 생성
3. greedy 답변 코드 실행 및 partial pass rate 계산
4. 고정 문제 프롬프트 로그확률 계산
5. CDD 계산
6. 문제 기반 Perplexity와 Min-k 계산
7. 생성 답변 기반 Completion Perplexity와 Completion Min-k 계산
8. 원자료와 detector score 저장

Mock dry run도 같은 의미의 데이터를 만들도록 동일하게 변경했다.

## 7. BF16 기준선 확정

기존에는 설정이 다음과 같았다.

```python
Quant.FP16 = "fp16"
```

반면 실제 로더는 `torch_dtype="auto"`를 사용해 배포 checkpoint가 BF16이면 실제로는
BF16으로 로드했다. 이 경우 설정·manifest에는 FP16, 실제 실행은 BF16으로 기록될 수
있었다.

이제 기준선을 다음처럼 명시적으로 고정했다.

```python
Quant.BF16 = "bf16"

AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)
```

따라서 설정 이름, 실제 모델 dtype, manifest 표기가 모두 BF16으로 일치한다.

확인한 공개 checkpoint config는 다음과 같다.

| 모델 | 배포 dtype |
|---|---|
| Qwen2.5-7B-Instruct | BF16 |
| Qwen2.5-32B-Instruct | BF16 |
| Olmo3-7B-Instruct | BF16 |

Qwen2.5-7B를 H100에 실제로 로드해 `model.dtype`과 첫 parameter dtype이 모두
`torch.bfloat16`임을 확인했다.

최종 양자화 사다리는 다음과 같다.

```text
BF16 → BNB INT8 → BNB NF4 → AWQ INT4
```

## 8. Manifest 변경

실행 manifest의 hash 대상 설정에 다음 필드를 추가했다.

```json
{
  "baseline_dtype": "bfloat16"
}
```

따라서 BF16 기준선 여부가 실행 설정 hash에 포함되며, 과거의 FP16 명칭 실행과 동일한
설정으로 취급되지 않는다.

## 9. 실행 스크립트와 문서 동기화

다음 실행 스크립트의 기준선을 `Quant.BF16`으로 변경했다.

- `pipeline/scripts/run_pilot.py`
- `pipeline/scripts/run_main.py`

영어 정본과 한국어 미러도 함께 수정했다.

- `paper/paper_draft.md`
- `paper/paper_draft_ko.md`

주요 문서 변경은 다음과 같다.

- 모델 표의 기준 정밀도를 BF16으로 통일
- 컴퓨트 표의 16-bit 열을 BF16으로 변경
- 확증 대비를 `BF16 → BNB-NF4`로 변경
- §4.3 양자화 사다리를 BF16 기준으로 확정
- `BF16 또는 FP16`이라는 모호한 표현 제거
- 실제 GPU 로더가 미구현이라는 낡은 설명 제거

다음 문서와 참고 자료도 동기화했다.

- `pipeline/README.md`
- `pre_pilot_next_steps.md`
- `pipeline_build_plan.md`
- `pipeline_implementation_log.md`
- `reference/experiment_design.csv`
- `reference/experiment_models.csv`
- `reference/word_dict.md`

과거 설계 변경을 기록하는 `paper/revision_provenance.md`는 당시의 역사적 FP16 표현을
보존했다.

## 10. 테스트 및 실제 H100 검증

새로 고정한 회귀 조건은 다음과 같다.

- 생성 없이 `score_prompt_logprobs()`를 호출할 수 있음
- chat wrapper 토큰이 prompt 점수에서 제외됨
- mock 모델도 고정 문제 API를 구현함
- 실제 `run()`이 prompt API를 주 탐지기 계산에 사용함
- completion confidence가 별도 이름으로 저장됨
- parquet에서 prompt/completion 로그확률 배열이 round-trip됨
- BF16 로더가 `dtype=torch.bfloat16`을 강제함
- manifest hash에 `baseline_dtype`이 포함됨
- 기존 CDD, pass rate, cache, 분석 경로가 깨지지 않음

전체 테스트 결과:

```text
134 passed
6 warnings
```

경고는 evalplus sandbox 테스트에서 Python multiprocessing의 `fork()` 사용과 관련된 기존
deprecation warning이며 이번 변경의 실패가 아니다.

실제 H100 smoke test 조건:

```text
GPU: NVIDIA H100 80GB
모델: Qwen2.5-7B-Instruct
정밀도: BNB-NF4
문항: HumanEval 5개
생성: greedy 1개 + temperature 0.8 sample 2개
```

결과:

```text
Peak GPU memory: 6.69GB
모든 smoke-test 체크 통과
```

생성된 실제 원자료도 확인했다.

```text
generations.parquet: 15행
detector_scores.parquet: 25행
prompt_token_logprobs가 저장된 greedy 행: 5개
```

## 11. 변경하지 않은 범위

이번 업데이트에서는 다음을 변경하지 않았다.

- 논문의 Q1/Q2 구조
- CDD 공식과 CDD 샘플 수
- pass@1 및 partial pass rate 계산
- contamination proxy 라벨
- TRACER 설계
- 통계 공식과 검정력 계산
- 모델 로스터
- LiveCodeBench cutoff
- 파일럿 표본 수
- 공식 정답과 생성 답변의 verbatim 유사도라는 새로운 연구 질문

## 12. 2026-08-20 당시 완료 상태

완료된 plan 단계:

1. 고정 문제 프롬프트 채점 및 completion confidence 분리
2. BF16 기준선 확정과 코드·manifest·논문 동기화

당시 남아 있던 단계:

1. 실제 parquet 파일럿 집계기 구현
2. `pilot_summary.json`과 `power_recompute.json` 생성
3. LiveCodeBench stdin/functional 실제 출력 smoke test
4. LCB 표본 수와 Q2 분석 지위의 최종 문서 반영
5. Olmo3 날짜 경계와 corpus ground truth 구축
6. TRACER 재구현 검증과 proxy-label 오류율 측정
7. Qwen2.5-7B 및 Olmo3-7B BF16→BNB-NF4 파일럿
8. CDD 관문 판정과 본 실험 규모 동결
9. 다섯 모델 × 네 정밀도 본 실험

## 13. 별도 확인된 실행 리스크

- Llama-3.1-8B-Instruct Hugging Face 저장소는 인증이 필요하다.
- Olmo 32B arm은 유효하지 않았던 `allenai/Olmo-3-32B-Instruct` 대신 공식 최종 RLVR
  Instruct 모델 `allenai/Olmo-3.1-32B-Instruct`로 교체했다.
- 기존 설계의 7B–32B 크기 축은 유지한다.
