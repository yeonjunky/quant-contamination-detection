# AGENTS.md — 프로젝트 지침

## 1. 이 프로젝트가 무엇인가

논문 하나를 준비하는 저장소다.

> **Does Quantization Erase the Evidence? Contamination-Detection Signals Under
> Post-Training Quantization in LLMs on Code Generation**
>
> (가제 — 정본 `paper/paper_draft.md` 제목을 따른다)

주장의 씨앗: 양자화로 인한 정확도 하락은 관행적으로 *능력* 손실로 해석되는데, 그 일부는
사실 *암기된 벤치마크 답*의 손실일 수 있다. 이 가설을 검증 가능한 형태로 좁힌 것이 Q1/Q2다.

- **Q1 (주 질문).** 학습 후 양자화(PTQ)가 첨도 기반 탐지기(CDD)와 확률 기반 탐지기
  (perplexity, Min-k% Prob)를 **서로 다르게** 변조하는가.
  - Q1a: 정밀도 간 문항별 탐지기 점수 이동 (대응 비교, **오염 라벨 불필요**)
  - Q1b: 정밀도별 `possible-exposure`/`shared-clean-control` 대리 라벨 분리 성능(AUC) 변화와
    탐지기 계열 간 순위 역전. 실제 오염 ground truth AUC로 해석하지 않음
- **Q2 (보조 질문).** pass@1에 양자화 × 노출 대리변수 교차효과가 있는가. (검정력 부족으로 강등됨)

**현재 상태: 실행 전(pre-execution).** 어떤 실험도 아직 돌리지 않았다. 문서의 `[TBD]` 자리는
전부 §5 실행 후에 채울 것이다. **실험 결과를 지어내지 말 것.** 수치가 필요하면 계산 근거를
명시하고, 계획값인지 측정값인지 항상 구분해서 쓴다. 로컬 드라이런과 H100 스모크 테스트는
구현 검증일 뿐이며, 그 출력은 논문 결과·검정력 재산정·탐지기 선별에 사용하지 않는다.

---

## 2. 파일 지도 — 무엇이 정본인가

디렉토리는 지위(정본/사료/폐기/참고)별로 나뉘어 있다. `AGENTS.md`는 항상 루트에 둔다.

| 파일 | 지위 | 취급 |
|---|---|---|
| `paper/paper_draft.md` | **영어 정본 (canonical)** | 모든 수정은 여기서 시작 |
| `paper/paper_draft_ko.md` | 한국어 미러 | 영어 수정 시 **반드시 동시 반영**. 충돌 시 영어 우선 |
| `paper/revision_provenance.md` | 감사 추적 | 논문 본문에서 분리한 개정 이력. 투고 원고에 포함 안 함 |
| `review/review_findings*.md`, `review/review_response.md` | 검토 기록 (1~5·7~9차) | 읽기 전용 사료. 수정하지 말 것. 최신은 `review_findings_round9.md` |
| `superseded/topic3_experiment_plan.md`, `superseded/contamination_literature_review.md` | **SUPERSEDED** | 정본에 통합·대체됨. **인용 금지** (배너 참조) |
| `reference/word_dict.md` | 비전문가용 용어 해설 | 9차 검토까지 반영 (2026-09-18 갱신). 정본 수정 시 여기 용어도 함께 볼 것 |
| `reference/contamination_literature.csv`, `reference/experiment_design.csv`, `reference/experiment_models.csv` | 문헌·실험 설계 목록 | `contamination_literature.csv`는 일부 제목이 축약/부정확 — 투고 전 arXiv 대조 필요 |
| `pipeline/` | 데이터 수집·분석 코드 | 확증 검정 `src/qcd/analysis/confirmatory.py`, §4.5.5 구간 `conditional_logit.py`, 포함률 검증 `coverage.py` + `scripts/verify_interval_coverage.py`, 분석 진입점 `scripts/run_analysis.py`. manifest의 `study_phase`가 검증 출력과 본 실행 데이터를 갈라 놓는다 |
| `pdfs/2*.pdf` | 원문 PDF 7편 | **`.gitignore`에 `*.pdf`가 있어 git에 없다** (§6 참조) |
| `figures/fig_*.png` | 검정력·관문 그림 | 표 수치를 고칠 때 그림과 어긋나지 않는지 확인 |

---

## 3. 작업 규율

아래 규칙은 이전 검토에서 확인된 오류의 재발을 막기 위한 것이다.

### 3.1 인용은 원문 PDF와 자구 대조하라

기억이나 요약본에 의존하지 않는다. 이전 오류:

- 효과 크기를 **날조**했다 (arXiv:2505.20276을 "1–4%p"로 오인용; 실제는 0.8%/59%, 게다가
  코드 생성이 아니라 **롱컨텍스트** 평가 결과였다).
- 원문 초록의 **조건이 붙은 한정 문구**를 강한 주장으로 읽었다
  ("outperform CDD in all conditions where any method exceeds chance"는 확률 기반 기법이
  항상 우연을 넘는다는 뜻이 아니다).
- 원문이 "should not be extrapolated to larger scales"라고 쓴 문장을 **인용해 놓고**
  바로 다음 문단에서 7B→70B 외삽을 했다.

대조 방법: `python3 -c "import pdfplumber; ..."` 로 텍스트 추출 후 grep. PDF 텍스트는
공백이 뭉개져 나오므로(`trainableparameters`) 검색어에서 공백을 빼고 찾아라.

### 3.2 검토 의견은 반영 전에 검증하라

5차 검토의 4건 중 1건이 틀렸는데, 검증 없이 반영되어 **새 오류가 정본에 들어갔다.**
(§4.2의 "~4M vs 3–25M" 사례.) 검토 의견을 받으면 **채택 전에 독립 재계산 / 원문 재대조**를
하고, 기각한 항목은 기각 사유를 남긴다.

### 3.3 숫자는 재계산하고 계산 경로를 기록하라

반복된 실패: 서로 다른 가정으로 계산된 수치를 같은 표에 섞어 놓기. 사례 —

- 라벨 잡음 표에서 e=10% 행만 "SE 고정" 방식, 나머지는 "SE도 감쇠" 방식으로 계산 (265 vs 287)
- 기저율 유도 상관 r을 p=0.5 표본 수 공식에 곱하기 → **척도 혼합 오류**
  (785와 ≈412는 다른 경로의 다른 숫자다. 합성되지 않는다)
- Q2 분해 표 4행을 분산 평가점과 β 가정이 섞인 "≈415–419"로 적기 → 같은 규약이면 **412**
  (9차 검토 D-S2)

표를 쓸 때는 **모든 행이 같은 방법으로 계산되었는지** 확인하고, 가정을 표 캡션에 명시하라.

### 3.4 수정 후 원본과 미러를 확인하라

2차 검토에서 적발된 패턴: 대응 문서(response)만 갱신하고 원본 서술 문서는 그대로 둠.
그리고 수정 과정에서 새 오류(잘린 arXiv ID, 깨진 그림 참조)를 만들었다.
수정 후에는 반드시 grep으로 영/한 양쪽 반영과 수치 일치를 확인한다.

### 3.5 논문 본문에 자기 정정 서사를 쓰지 말라

"이전 판은 ~라고 썼는데 틀렸다", "그 논증을 철회한다" 같은 서술은 논문에 들어가지 않는다.
**정정으로 얻은 결론(근거)만 남기고 프레이밍은 논문 화법으로 바꾼다.**

- ✗ "5차 검토가 X를 권고했으나 기각한다"
- ✓ "~4M과 3–25M은 서로 대체 가능하지 않다. 7B 수치는 Table 1이 아니라 Discussion 본문에 있다"

이력이 필요하면 `paper/revision_provenance.md`로 보낸다.

---

## 4. 검증된 값

아래 값은 원문 PDF 대조 또는 독립 재계산으로 확인했다. 변경하려면 먼저 재검증한다.

### 4.1 검정력 / 표본 수 (α=0.05 양측, 80% 검정력 → 승수 2.8016)

| 항목 | 값 |
|---|---|
| Q1a 필요 문항 수 | d=0.3 → 87, d=0.2 → 196 |
| Q1b SE(AUC) @ n=164, AUC=0.70 | 0.0287 |
| Q1b 검출 한계 @ n=164, r=0.8 | **0.0509** (목표 0.05에 미달 → 근 n=170.005 → 약 **170**, 엄밀한 올림 **171**) |
| Q2 필요 n (10pp, 비페어링 p=0.5) | **785** ← 독립 셀·p=0.5·원시 %p 정규근사 계획 기준값 |
| Q2 필요 n (페어링 + 실제 기저율) | **≈412** ← 가정이 성립할 *경우*의 값. 계획 기준 아님. 785 대비 −47.5% |
| Q2 주 LCB 대비 MDE (비페어링 p=0.5) | 공통 대조 n=182, 의심 n→∞ → **14.7%p**; 의심 n=873 → **16.1%p**. 실제 arm별 의심 n은 873 이하이며 Olmo는 873 |
| Q2 보조 HumanEval 대비 MDE | HumanEval n=164, 대조 n→∞ → **15.5%p** |
| LCB release_v6 모델별 주 라벨 수 | Qwen 690/183/182, Llama 326/547/182, Olmo 873/0/182 (`possible`/`clean-by-model-cutoff`/`shared`) |
| 구현 검증 경계 | 드라이런·스모크 출력은 연구 데이터가 아니며 CDD 자격, C1–C4, 표본 수 또는 검정력을 바꾸지 않음 |
| §4.5.3 분해 표 3행 (σ=1.5 함의 상관) | r = **0.293089** (수치 적분; MC로는 0.291~0.293이 나옴 — 적분값이 정본), n = 785×(1−r) = **555** |
| 확증 검정 사이징 (§4.5.6, Holm α/4) | 승수 **3.339**, Q1a d=0.3 → **≈124**, d=0.2 → **≈279** (C4는 이 공식으로 사이징되지 않음) |

§4.5.3 분해 표의 네 행(785 / 557 / 555 / ≈412)은 **한 가지 규약**을 공유한다: 이항 분산은 하락 없는
귀무 기저율에서 평가하고, 페어링 감소는 그 점에서 σ=1.5 문항 난이도 모형이 함의하는 정밀도 간 상관
(p=0.5에서 0.293089, 0.85에서 0.221737, 0.35에서 0.282597)을 쓰며, 기저율 0.85/0.35는 난이도 분포에
대한 **주변 평균**이다. 4행을 대립가설 쪽 분산에서 평가하면 415(β=0.25)–426(β=1.0)이 나온다 —
이전 판의 "≈415–419"는 β 값과 분산 평가점이 섞인 범위였다 (9차 검토 D-S2). 계획 기준값은 785로 불변.

실제 양성 비율 0.5와 점수에 독립적인 대칭 라벨 반전을 가정한 감쇠: `AUC_obs − 0.5 = (1−2e)(AUC_true − 0.5)`, 따라서 ΔAUC도 ×(1−2e)로 감쇠.
필요 문항 수 (ΔAUC_true=0.050, r=0.8):

| e | SE 고정 (AUC=0.70) | **SE도 감쇠 (정본 채택)** |
|---|---|---|
| 0% | 170 | **170** |
| 10% | 265 | **287** |
| 20% | 471 | **541** |
| 30% | 1,060 | **1,268** |

표는 반올림한 계획 근사값이다. 오른쪽 열의 엄밀한 정수 올림은 171/288/542/1,268이다. Q1a 비보정 87/196도 근사값이며 정수 올림은 88/197이다. 정본은 **오른쪽 열만** 쓴다. `figures/fig_round4_corrections.png` 패널 b도 오른쪽 열 기준이다.

### 4.2 arXiv:2603.03203 (*No Memorization, No Detection*, Sela) — 자구 확인 완료

- CDD = **Contamination Detection via output Distribution**, **Dong et al. (2024)** 도입.
  Sela는 CDD의 저자가 아니라 **재현 연구자**다. 귀속을 혼동하지 말 것.
- 대상 규모: Pythia **70M–410M**. Table 1은 이 세 크기만 수록 (98K–405M).
- **오염 주입 방식은 LoRA만이 아니다.** 원문은 GSM8K/HumanEval/MATH에 3·20 epoch로 오염을 주입하며
  미세조정 방식 세 가지를 쓴다: **LoRA r=8, LoRA r=256, 전체 미세조정**
  (원문 서론: *"fine-tuning method (LoRA with rank 8 and 256, and full fine-tuning)"*; §4 설정에서
  r=8은 파라미터의 0.1–0.2%, r=256은 4–6%, 전체 미세조정은 100%).
  CDD가 실제로 작동하는 조건은 **전체 미세조정**이다 (Pythia-410M, GSM8K, 오염 수준 10, 3 epoch에서
  CDD 정확도 **0.955**; 같은 조건의 LoRA r=8은 우연 수준).
- **CDD 실패 요인** (정본 §2.4의 현재 서술과 같게 유지): CDD는 미세조정이 **verbatim 암기**를 만들 때만
  작동한다 — *"CDD's effectiveness depends critically on whether fine-tuning produces verbatim
  memorization"*, *"CDD requires output distribution collapse to succeed"*. 그 붕괴 여부는 문턱이 있는
  현상이다: *"CDD accuracy transitions sharply from chance to >90% as fine-tuning capacity crosses a
  threshold"*이며 문턱은 모델 크기·어댑터 rank·학습 길이의 상호작용에 달려 있다. 원문 자신의 귀속은
  *"The relevant factor is not the LoRA rank itself but the absolute number of trainable parameters."*
  다만 **이 결과를 우리 모델에 대한 계단 함수로 쓰지 않는다** (정본 §4.5.1: 소형 모델 결과는
  step function도 안정적 효과 크기도 입증하지 않는다).
- **파라미터 수치 — 이게 가장 많이 틀린 지점이다:**
  - **~4M** = 7B 모델의 LoRA **r=8**. 원문 §5 Discussion 본문에 자구 그대로 있다
    (*"LoRA r=8 on a 7B model yields roughly 4M trainable parameters"*). Table 1에는 없다.
  - **3–25M** = 재현 논문 **자신의** r=256 구성 범위 (Table 1: 3.1M / 9.4M / 25.2M).
  - **둘은 다른 양이다. 대체하지 말 것.**
- 임계값 ξ: 원 CDD 논문은 7B에서 교정한 ξ=0.01 고정. 재현 논문은 **평가셋에서 Youden index
  최대화**로 재선택하며 스스로 *"This gives CDD every advantage"*라고 밝힌다 → 낙관 편향된
  oracle 임계값. **AUC는 임계값 무관이므로 Q1b에는 ξ 재교정이 불필요하다.**
- 확률 기반 vs CDD — 강도별 문장 3개 (섞지 말 것):
  - 초록(조건부 주장 — 동어반복이 아니다. Table 2 기준 우연 초과는 CDD 7/27, perplexity 26/27, Min-k% 25/27): *"outperform CDD in all conditions where any method exceeds chance"*
  - 결론(더 강함): 위 문장 + *"including those where CDD fails entirely"*
  - **정본이 채택한 인용**: *"The gap is largest precisely where it matters most: at low
    contamination levels and under parameter-efficient fine-tuning, where CDD is uniformly at
    chance but probability-based methods already show signal."* (§4.2 Results 절. **Limitations
    절이 아니다** — 예전에 잘못 라벨링했던 이력이 있다)
- 자기 한계: *"should not be extrapolated to larger scales without further investigation"*.
  → **32B에서 CDD가 작동하는지는 미지**이며, 우리 설계는 이를 가정이 아니라
  고정된 본 연구의 경험적 결과로 다룬다(§4.7). 구현 검증 결과로 CDD를 선별하지 않는다.
  (2026-08-05 설계 변경으로 70B arm이 제거되어 상한이 32B다.)

### 4.3 그 밖의 인용 주의

- **arXiv:2505.20276** — 8bit ≈0.8% 하락 / 4bit 최대 59% 하락, Llama-3.1-70B BNB-nf4 32% 하락.
  **코드 생성이 아니라 롱컨텍스트(>64K) 평가 결과다.** BNB-nf4 arm의 "최악 사례" 사전 정보로만
  인용할 것. 측정 대상은 Llama-3.1-**70B**이고 우리 arm은 Llama-3.1-**8B** — 같은 계열이지만
  약 9× 작으므로 **규모 불일치 사전 정보**다 (§6 타당성 위협에 명시됨).
- **arXiv:2410.16454** — 4bit 양자화 후 잔존 지식 21%→83%. 정식 제목 확인 완료 (7차 검토):
  *"Catastrophic Failure of LLM Unlearning via Quantization"* (ICLR 2025). 원문의 한정어
  "for unlearning methods **with utility constraints**"에 유의 — 무제약 수치는 원문 스스로
  misleading이라 경고한다.
- **arXiv:2605.15138** (*Forgetting That Sticks*) — unlearning 파라미터 변화가 NF4 bin 폭보다
  47–828× 작다.
- 2026-09-09 제목 확인: arXiv:2409.09927 = *Towards Data Contamination Detection for Modern Large Language Models: Limitations, Inconsistencies, and Oracle Challenges*; arXiv:2505.20276 = *Does quantization affect models' performance on long-context tasks?*.
  **2026-08-09 해결:** arXiv:2311.04850 = *Rethinking Benchmark and Contamination **for Language Models**
  with Rephrased Samples* (Yang, Chiang, Zheng, Gonzalez, Stoica), arXiv:2403.04811 =
  *Quantifying Contamination in **Evaluating** Code Generation **Capabilities of Language Models***
  (ACL 2024) — 둘 다 arXiv abs 페이지 직접 대조. 정본 References 반영 완료.
  **투고 전 arXiv에서 정식 제목 확인 필요** (본문에 액션으로 남겨 둠). 2410.16454는 위와 같이
  해결됨. 2605.24079(TRACER)와 2511.12116(LLMLagBench)의 정식 제목도 7차 검토에서 PDF로 확인됨
  (`review/review_findings_round7.md` 부록 A 및 2.5 참조).

---

## 5. 설계상 지켜야 할 논증 구조

건드리면 논문 전체가 무너지는 구조적 결정들이다. 바꾸려면 근거를 갖고 명시적으로 하라.

1. **Q1이 주, Q2는 보조.** 이유는 검정력이다. 주 Q2 대비는 모델별 LCB
   `possible-exposure` 대 182문항 `shared-clean-control`이며, 보수적 비페어링 p=0.5 계산에서
   의심 셀이 무한해도 MDE가 14.7%p, 최대 873문항 외피를 모두 써도 16.1%p다. HumanEval과
   MBPP+는 풀링하지 않는 별도 탐색적 Q2 대비이며, HumanEval의 최선 MDE는 15.5%p다.
   **Q2의 유의성에 논문의 기여 주장을 걸지 말 것.**
2. **Q1a는 노출 라벨 없이 분석한다.** Q1b의 추정 대상은 시간 대리 라벨 AUC다.
   `no-match-found`는 검증된 비노출이 아니므로 어떤 arm에서도 오류율 *e*를 실측하지 않는다.
   *e* 표는 실제 오염 AUC로 재해석할 때의 구성 타당도 민감도 분석일 뿐이며 Q1b 사이징의
   전제 조건이 아니다.
3. **Q2는 조건부 로그오즈 스케일로.** 하락 대비 J와 회귀 계수의 부호를 구분한다.
   Q=1 양자화, E=1 노출 가능 코딩에서 β_QE=−J이다. 기저율 0.85/0.35 예제의 원시 %p
   아티팩트는 음수지만 보편적 부호 규칙은 아니다. 난이도 계층화는 필수 진단이며 비유의성으로
   오즈비 일정 가정을 입증하지 않는다.
4. **Q2의 어느 부호도 교란에서 자동으로 보호되지 않는다.** 보고 구간은 **문항 층화 조건부 로지스틱
   회귀의 β_QE Wald 구간**이다(모델별 적합, 문항 절편은 조건부 우도에서 소거). **변분 Bayes 사후 SD로
   구간을 만들지 않는다** — mean-field 근사는 계수 간 사후 상관을 버려 구간이 명목 포함률을 못 낸다.
   `fit_vb`는 점추정과 분산 성분에만 쓴다. 본 실행 전에 설계와 같은 모양의 합성 자료(690/182, 난이도
   SD 1.5, 알려진 β_QE)로 **포함률을 검증**하고 그 결과가 어느 방법을 보고할지 정한다 — 본 실행 출력이
   정하지 않는다. 측정값(200회 반복, `pipeline/analysis_artifacts/interval_coverage_check.json`):
   조건부 로지스틱 **0.950**, 변분 Bayes **0.485** (명목 0.95). 사전 동등성 마진이 없으므로 null을
   동등성으로 해석하지 않는다.

5. **HumanEval과 MBPP+를 풀링하지 말 것.** 난이도 분포가 달라 단일 조건 *내부*로 기저율 교란이
   재유입된다. n=542는 참고값일 뿐 분석 셀이 아니다.
6. **관찰 연구지 인과 연구가 아니다.** 기성 사전학습 모델에는 오염을 무작위 배정할 수 없다.
   결과는 **연관(association)**으로 보고한다.
7. **QAT 배포 모델(Gemma 계열 포함)은 전면 제외.** (2026-08-05 결정 — 이전에는 Gemma-4-31B-it을
   부록 QAT-vs-PTQ 비교용으로 유지했으나 완전 제외로 변경. 공식 QAT 체크포인트 / thinking 모드 /
   멀티모달 3종 교란에 더해, q4_0 포맷이 제2 추론 스택을 요구해 그 numerics 차이가 QAT-vs-PTQ
   대비 자체를 교란한다. QAT 비교는 future work. `paper/revision_provenance.md` 참조.)
8. **아키텍처는 통제 축이 아니다.** dense GQA+RoPE를 공유해도 전체 아키텍처·토크나이저·
   학습 구성이 같다는 뜻은 아니다. 크기는 기술적 비교 축이며 계열 간 인과 효과로 해석하지 않는다.
   코퍼스 투명성은 Olmo에서 양성 근거를 확보하는 운영상 선택 속성이다. 로스터는
   Qwen2.5-7B/32B, Olmo3-7B/Olmo3.1-32B, Llama-3.1-8B, 전부 Instruct로 유지한다.
   Llama의 LLMLagBench 결과는 독립 지식 경계 진단이지 학습 데이터 cutoff의 검증이 아니다.
   Olmo 코퍼스 검색은 사전학습과 post-training 모두를 포괄한다.

9. **구현 검증과 연구 데이터를 섞지 말 것.** 로컬 드라이런과 제한된 H100 스모크 테스트는
   모델 로딩·스키마·로그확률·샌드박스·메모리·처리량만 확인한다. 그 출력으로 효과 크기, AUC,
   pass rate, 상관, 검정력 또는 CDD 자격을 판단하지 않는다. 결과 확인 전 운영 실패로 설정을
   바꿨다면 기록 후 다시 고정하고, 본 실행에서만 연구 데이터를 생성한다.
10. **확증 검정군은 §4.5.6의 4개뿐** (Qwen2.5-32B-Instruct의 C1–C3: 전체 LCB 1,055문항 bf16→nf4 이동; C4: 690/182 대리 집단에서 확률 AUC 두 개의 동일 가중 평균 대 CDD 순위 역전), 군내 Holm 보정. **그 외 전부 탐색적** — 결과를 본 뒤 확증군에 검정을 추가하거나 역할을
   바꾸는 것은 금지. CDD가 본 연구에서 우연 수준이어도 C4를 사후 제거하지 않는다. 추정 불가
   (라벨 집단이 빔, 점수 누락, gap 분산이 0)로 판정된 슬롯도 **p=1로 남겨** 다중성 계산에 유지한다.
   확증 검정 사이징은 α/4 승수 3.339 기준 (Q1a d=0.3 → ≈124문항)이며 **C4는 이 공식이 사이징하지
   않는다** — C4 정밀도는 두 gap과 여섯 AUC의 결합 공분산에 달려 있다 (§4.5.6의 예시 계산).

---

## 6. 작업 관행

- **영/한 동시 수정.** `paper/paper_draft.md`를 고치면 `paper/paper_draft_ko.md`도 같은 턴에 고친다.
  끝나면 핵심 수치 개수를 grep으로 대조한다.
- **verbatim 인용은 한국어판에서도 영어 원문 그대로 둔다** (인용 지위 보존). 번역 주에 명시됨.
- **PDF는 git에 없다.** `.gitignore`의 `*.pdf`는 하위 디렉토리도 포함해 매칭되므로 `pdfs/` 안의
  7편 모두 추적되지 않는다. 새 환경에서 인용 대조가 필요하면 arXiv에서 다시 받아 `pdfs/`에 넣는다.
  파일명: `2310.10628` `2310.16789` `2410.16454` `2511.12116` `2603.03203` `2605.15138` `2605.24079`.
  2단 조판 PDF는 좌/우 컬럼 분리 추출 필수 — 통짜 추출은 긴 인용 검색이 조용히 실패한다
  (7차 검토 §3.1).
- **bash 경로 주의.** 파일 도구는 `/Users/kim/Desktop/knowlodge-rot-by-quantization/`,
  bash는 `/sessions/<session>/mnt/knowlodge-rot-by-quantization/`.
- **scipy 없음.** 수치 계산은 numpy + 직접 구현(이분법 등)으로. Hanley–McNeil SE 공식과
  이분 탐색이면 위 표는 전부 재현된다. 단 `pipeline/.venv`에는 numpy와 scipy가 있으므로
  (`pipeline/.venv/bin/python`) §4.1 재검산은 그쪽에서 수치 적분으로 해도 된다.
- 폴더명 `knowlodge-rot-by-quantization`의 오타는 의도된 것이 아닐 수 있으나 그대로 둔다.

---

## 7. 다음 단계 (§5 실행 계획 요약)

번호는 **논문 §5의 단계 번호와 같게 유지한다.** (적대적 검토는 §5의 단계가 아니다 — 7·8·9차 모두
완료·반영했고 기록은 `review/review_findings_round7|8|9.md`와 `paper/revision_provenance.md`에 있다.)

1. 연속형 채점 파이프라인 (부분 테스트 통과율 + 토큰 로그확률) — Q1의 **전제 조건**.
   부분 점수 자체는 탐색적 지표
2. 탐지기 채점 파이프라인 (CDD / perplexity / Min-k%). CDD의 문항당 다중 샘플 비용을 견적에 반영,
   1번과 기저 생성물 공유
3. 모델–문항 시간 라벨을 실체화하고 센다. LiveCodeBench `release_v6`, 공통 경계 2025-01-01에서
   pre 873 / 공통 대조 182 / 전체 1,055. **의심 셀은 arm별로 다시 세며 모든 모델에 873을 쓰지 않는다.**
   각 ≥1,000 목표 미달이므로 Q2는 보조·구간 중심 분석
4. 모델별 시간 경계를 검증·고정한다: Llama-3.1-8B는 주 2024-01-01(선언 2023-12), 민감도
   2023-04-01(LLMLagBench 탐지 2023-03; 실제 학습 cutoff 검증은 아님). Qwen2.5는 공식 cutoff 선언이
   없어 출시 다음 날 2024-09-20. Olmo3-7B/Olmo3.1-32B 모델 카드는 모두 `Date cutoff: Dec. 2024`를
   적어 **공통 LCB-post 시작일은 2025-01-01**. **경계가 상·하한으로 나뉘는 arm은 경계 양쪽으로
   Q1b/Q2 민감도 재분석** (선언=주/탐지=민감도 사전 고정, 라벨만 변경이라 추가 GPU 비용 0; Llama의
   탐지 경계 LCB 노출 가능 문항은 0이므로 해당 대비는 추정 불가; 모호 창 탐지기 점수 비교는 Q1b
   라벨로 역주입 금지 — §4.2). 코퍼스 검색 결과는 시간 라벨이 아니라 별도 코퍼스 축에 넣는다
5. TRACER(arXiv:2605.24079)로 Olmo3 사전학습·post-training 코퍼스에서 잔여 오염의 확인 양성을
   탐색한다. 공개 코드 없음이 확인되어 재구현 필요(프롬프트·임계값은 원문 부록에 공개, 7차 검토
   부록 A 참조)이며, **사전학습 코퍼스 적용은 원문이 검증한 범위 밖의 확장**이다. 개방 코퍼스 탐지
   3계열(문자열 매칭 / 표면·구조 프로그램 매칭 / 패러프레이즈)을 모두 돌리고 `confirmed-match`,
   `no-match-found`, `not-observable` 수를 **계열별로 따로** 보고한다. `no-match-found`를 음성 정답으로
   쓰지 않는다. 6번과 병행 가능
6. 구현 검증만 (§4.6): 로컬 합성 드라이런과 제한된 H100 스모크 테스트로 로딩, 양자화 호환성, 스키마,
   유한 로그확률, 샌드박스, 메모리와 처리량만 확인한다. 출력은 연구 데이터와 분리하고 효과 크기,
   AUC, pass rate, 탐지기 순위 또는 검정력을 계산하지 않는다
7. **분석을 구현하고 합성 자료로 검증한 뒤 운영 구성을 고정한다.** 본 실행 관측이 생기기 전에
   C1–C3 대응 t검정, C4의 여섯 AUC DeLong 공분산·gap 대비·교집합–합집합 p값·Holm 보정,
   §4.5.5의 모델별 구간 절차를 작성한다. 알려진 생성 모수의 합성 자료로 C1–C4 회복과 후보 구간
   방법별 달성 포함률을 재현 횟수와 함께 기록한다. **그 검증이 보고 구간 방법을 정한다 — 본 실행
   출력이 정하지 않는다.** 그 뒤 모델·문항·채점·분석 구성을 고정한다. 검증 자료는 C1–C4, 계획 문항
   집합, 탐지기 우선순위를 바꾸지 않는다
8. **본 실행 — 연구 데이터의 유일한 출처.** 조건마다 **문항 단위 원자료 전부 저장** (pass@1,
   부분 점수, 토큰 로그확률, 탐지기 3종 점수). 집계값만 저장하면 대응·혼합효과 분석이 불가능해진다
9. **분석 — 7번에서 고정한 코드를 수정 없이 실행한다.** Q1a/Q1b는 고정 문항 집합에 고정 검정을
   돌리고 C4의 점수 방향·역전 기준을 그대로 쓴다. Q2는 β_QE와 하락 대비 J=−β_QE를 포함률 검증된
   구간과 함께 보고한다. 경계 민감도는 §4.2의 고정 라벨을 적용하고 빈 집단은 추정 불가로 보고한다
