# AGENTS.md — 프로젝트 지침

## 1. 이 프로젝트가 무엇인가

논문 하나를 준비하는 저장소다.

> **Does Quantization Erase the Evidence? Contamination-Detection Signals Under
> Post-Training Quantization in Code LLMs**

주장의 씨앗: 양자화로 인한 정확도 하락은 관행적으로 *능력* 손실로 해석되는데, 그 일부는
사실 *암기된 벤치마크 답*의 손실일 수 있다. 이 가설을 검증 가능한 형태로 좁힌 것이 Q1/Q2다.

- **Q1 (주 질문).** 학습 후 양자화(PTQ)가 첨도 기반 탐지기(CDD)와 확률 기반 탐지기
  (perplexity, Min-k% Prob)를 **서로 다르게** 변조하는가.
  - Q1a: 정밀도 간 문항별 탐지기 점수 이동 (대응 비교, **오염 라벨 불필요**)
  - Q1b: 정밀도별 오염/청정 분리 성능(AUC) 변화와 탐지기 계열 간 순위 역전
- **Q2 (보조 질문).** pass@1에 양자화 × 오염 교차효과가 있는가. (검정력 부족으로 강등됨)

**현재 상태: 실행 전(pre-execution).** 어떤 실험도 아직 돌리지 않았다. 문서의 `[TBD]` 자리는
전부 §5 실행 후에 채울 것이다. **실험 결과를 지어내지 말 것.** 수치가 필요하면 계산 근거를
명시하고, 계획값인지 측정값인지 항상 구분해서 쓴다.

---

## 2. 파일 지도 — 무엇이 정본인가

디렉토리는 지위(정본/사료/폐기/참고)별로 나뉘어 있다. `AGENTS.md`는 항상 루트에 둔다.

| 파일 | 지위 | 취급 |
|---|---|---|
| `paper/paper_draft.md` | **영어 정본 (canonical)** | 모든 수정은 여기서 시작 |
| `paper/paper_draft_ko.md` | 한국어 미러 | 영어 수정 시 **반드시 동시 반영**. 충돌 시 영어 우선 |
| `paper/revision_provenance.md` | 감사 추적 | 논문 본문에서 분리한 개정 이력. 투고 원고에 포함 안 함 |
| `review/review_findings*.md`, `review/review_response.md` | 검토 기록 (1~5·7·8차) | 읽기 전용 사료. 수정하지 말 것 |
| `superseded/topic3_experiment_plan.md`, `superseded/contamination_literature_review.md` | **SUPERSEDED** | 정본에 통합·대체됨. **인용 금지** (배너 참조) |
| `reference/word_dict.md` | 비전문가용 용어 해설 | 8차 검토까지 반영 (2026-08-09 갱신). 정본 수정 시 여기 용어도 함께 볼 것 |
| `reference/contamination_literature.csv`, `reference/experiment_design.csv`, `reference/experiment_models.csv` | 문헌·실험 설계 목록 | `contamination_literature.csv`는 일부 제목이 축약/부정확 — 투고 전 arXiv 대조 필요 |
| `pdfs/2*.pdf` | 원문 PDF 6편 | **`.gitignore`에 `*.pdf`가 있어 git에 없다** (§6 참조) |
| `figures/fig_*.png` | 검정력·관문 그림 | 표 수치를 고칠 때 그림과 어긋나지 않는지 확인 |

---

## 3. 작업 규율

아래 규칙은 이전 검토에서 확인된 오류의 재발을 막기 위한 것이다.

### 3.1 인용은 원문 PDF와 자구 대조하라

기억이나 요약본에 의존하지 않는다. 이전 오류:

- 효과 크기를 **날조**했다 (arXiv:2505.20276을 "1–4%p"로 오인용; 실제는 0.8%/59%, 게다가
  코드 생성이 아니라 **롱컨텍스트** 평가 결과였다).
- 원문 초록의 **동어반복적 한정 문구**를 강한 주장으로 읽었다
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
  (785와 ≈417은 다른 경로의 다른 숫자다. 합성되지 않는다)

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
| Q1b 검출 한계 @ n=164, r=0.8 | **0.0509** (목표 0.05에 미달 → 정확히 풀면 **n=170**) |
| Q2 필요 n (10pp, 비페어링 p=0.5) | **785** ← 가정 없는 계획 기준값 |
| Q2 필요 n (페어링 + 실제 기저율) | ≈415–419 ← 가정이 성립할 *경우*의 값. 계획 기준 아님 |
| CDD 파일럿 관문 (기저 AUC 손익분기) | **0.7936** (문서 표기 ≈0.79) |
| §4.5.3 분해 표 3행 (σ=1.5 함의 상관) | r = **0.293089** (수치 적분; MC로는 0.291~0.293이 나옴 — 적분값이 정본), n = 785×(1−r) = **555** |
| 확증 검정 사이징 (§4.5.6, Holm α/4) | 승수 **3.339**, Q1a d=0.3 → **≈124**, d=0.2 → **≈279** |

라벨 잡음 감쇠: `AUC_obs − 0.5 = (1−2e)(AUC_true − 0.5)`, 따라서 ΔAUC도 ×(1−2e)로 감쇠.
필요 문항 수 (ΔAUC_true=0.050, r=0.8):

| e | SE 고정 (AUC=0.70) | **SE도 감쇠 (정본 채택)** |
|---|---|---|
| 0% | 170 | **170** |
| 10% | 265 | **287** |
| 20% | 471 | **541** |
| 30% | 1,060 | **1,268** |

정본은 **오른쪽 열만** 쓴다. `figures/fig_round4_corrections.png` 패널 b도 오른쪽 열 기준이다.

### 4.2 arXiv:2603.03203 (*No Memorization, No Detection*, Sela) — 자구 확인 완료

- CDD = **Contamination Detection via output Distribution**, **Dong et al. (2024)** 도입.
  Sela는 CDD의 저자가 아니라 **재현 연구자**다. 귀속을 혼동하지 말 것.
- 대상 규모: Pythia **70M–410M**. Table 1은 이 세 크기만 수록 (98K–405M).
- **파라미터 수치 — 이게 가장 많이 틀린 지점이다:**
  - **~4M** = 7B 모델의 LoRA **r=8**. 원문 §5 Discussion 본문에 자구 그대로 있다
    (*"LoRA r=8 on a 7B model yields roughly 4M trainable parameters"*). Table 1에는 없다.
  - **3–25M** = 재현 논문 **자신의** r=256 구성 범위 (Table 1: 3.1M / 9.4M / 25.2M).
  - **둘은 다른 양이다. 대체하지 말 것.**
- 임계값 ξ: 원 CDD 논문은 7B에서 교정한 ξ=0.01 고정. 재현 논문은 **평가셋에서 Youden index
  최대화**로 재선택하며 스스로 *"This gives CDD every advantage"*라고 밝힌다 → 낙관 편향된
  oracle 임계값. **AUC는 임계값 무관이므로 Q1b에는 ξ 재교정이 불필요하다.**
- 확률 기반 vs CDD — 강도별 문장 3개 (섞지 말 것):
  - 초록(동어반복): *"outperform CDD in all conditions where any method exceeds chance"*
  - 결론(더 강함): 위 문장 + *"including those where CDD fails entirely"*
  - **정본이 채택한 인용**: *"The gap is largest precisely where it matters most: at low
    contamination levels and under parameter-efficient fine-tuning, where CDD is uniformly at
    chance but probability-based methods already show signal."* (§4.2 Results 절. **Limitations
    절이 아니다** — 예전에 잘못 라벨링했던 이력이 있다)
- 자기 한계: *"should not be extrapolated to larger scales without further investigation"*.
  → **32B에서 CDD가 작동하는지는 미지**이며, 우리 설계는 이를 가정이 아니라
  파일럿 관문(§4.6)으로 다룬다. (2026-08-05 설계 변경으로 70B arm이 제거되어 상한이 32B다.)

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
- 제목 미확인 상태로 남은 것: arXiv:2409.09927, 2505.20276.
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

1. **Q1이 주, Q2는 보조.** 이유는 검정력이다. Q2의 최소 검출 가능 효과는 청정 조건 데이터가
   무한해도 HumanEval 164문항 상한 때문에 15.5%p이며, 이는 양자화 문헌이 기대하게 하는 어떤
   효과보다 크다. **Q2의 유의성에 논문의 기여 주장을 걸지 말 것.**
2. **Q1a는 오염 라벨 없이 분석한다.**
   Q1b는 대리 라벨 품질(e)에 조건부다 → TRACER 잔여 오염 측정이 Q1b의 **전제 조건**.
3. **Q2는 로그오즈 스케일로.** 원시 %p로 하면 기저율 차이(0.85 vs 0.35)만으로 **항상 음수**인
   가짜 교차효과(−1~−3pp)가 생기는데, 하필 "양자화가 암기를 되살린다" 가설과 **같은 부호**다.
   난이도 계층화는 선택적 강건성 점검이 아니라 **필수 가정 검증**이다.
4. **Q2 결과의 해석은 비대칭이다.** 음수는 아티팩트와 같은 부호이므로 완화 조치 후에만
   해석하고, null은 사전 지정 동등성 마진이 있을 때만 해석한다.
5. **HumanEval과 MBPP+를 풀링하지 말 것.** 난이도 분포가 달라 단일 조건 *내부*로 기저율 교란이
   재유입된다. n=542는 참고값일 뿐 분석 셀이 아니다.
6. **관찰 연구지 인과 연구가 아니다.** 기성 사전학습 모델에는 오염을 무작위 배정할 수 없다.
   결과는 **연관(association)**으로 보고한다.
7. **QAT 배포 모델(Gemma 계열 포함)은 전면 제외.** (2026-08-05 결정 — 이전에는 Gemma-4-31B-it을
   부록 QAT-vs-PTQ 비교용으로 유지했으나 완전 제외로 변경. 공식 QAT 체크포인트 / thinking 모드 /
   멀티모달 3종 교란에 더해, q4_0 포맷이 제2 추론 스택을 요구해 그 numerics 차이가 QAT-vs-PTQ
   대비 자체를 교란한다. QAT 비교는 future work. `paper/revision_provenance.md` 참조.)
8. **아키텍처는 통제 축이 아니다.** Qwen2.5와 Llama-3.1은 둘 다 dense GQA+RoPE라 그 열은 아무
   정보도 담지 않는다. 모델 간 남는 설계 축은 **크기와 학습 코퍼스 투명성**이다. (2026-08-05: 70B 컴퓨트 제약으로
   Llama-3.3-70B가 제거되고 Llama-3.1-8B-Instruct로 대체됨 — LLMLagBench 리더보드에서 cutoff가
   기검증된 유일한 주 분석 모델. 최종 로스터: Qwen2.5-7B/32B, Olmo3-7B/32B, Llama-3.1-8B —
   전부 **-Instruct 변형** 확정 (2026-08-05, provenance (f)). 이에 따라 Olmo3 ground-truth
   검색은 사전학습 + post-training 셋 모두 포괄 (§5 5번).)
9. **CDD 파일럿 관문(기저 AUC ≥ 0.79)을 우회하지 말 것.** 실패 모드는 천장이 아니라 **바닥**이다.
   32B가 우연 수준 부근이면 효과 크기 자체가 0으로 붕괴하므로 **데이터를 늘려도 회복되지
   않는다.** 관문 결과는 그대로 보고한다.
10. **확증 검정군은 §4.5.6의 4개뿐** (C1–C3: bf16→nf4 Q1a 탐지기별 이동 @LCB, C4: Q1b 계열 순위
   역전), 군내 Holm 보정. **그 외 전부 탐색적** — 결과를 본 뒤 확증군에 검정을 추가하거나 역할을
   바꾸는 것은 금지. 확증 검정 사이징은 α/4 승수 3.339 기준 (d=0.3 → ≈124문항).

---

## 6. 작업 관행

- **영/한 동시 수정.** `paper/paper_draft.md`를 고치면 `paper/paper_draft_ko.md`도 같은 턴에 고친다.
  끝나면 핵심 수치 개수를 grep으로 대조한다.
- **verbatim 인용은 한국어판에서도 영어 원문 그대로 둔다** (인용 지위 보존). 번역 주에 명시됨.
- **PDF는 git에 없다.** `.gitignore`의 `*.pdf`는 하위 디렉토리도 포함해 매칭되므로 `pdfs/` 안의
  6편 모두 추적되지 않는다. 새 환경에서 인용 대조가 필요하면 arXiv에서 다시 받아 `pdfs/`에 넣는다.
  파일명: `2310.10628` `2410.16454` `2511.12116` `2603.03203` `2605.15138` `2605.24079`.
  2단 조판 PDF는 좌/우 컬럼 분리 추출 필수 — 통짜 추출은 긴 인용 검색이 조용히 실패한다
  (7차 검토 §3.1).
- **bash 경로 주의.** 파일 도구는 `/Users/kim/Desktop/knowlodge-rot-by-quantization/`,
  bash는 `/sessions/<session>/mnt/knowlodge-rot-by-quantization/`.
- **scipy 없음.** 수치 계산은 numpy + 직접 구현(이분법 등)으로. Hanley–McNeil SE 공식과
  이분 탐색이면 위 표는 전부 재현된다.
- 폴더명 `knowlodge-rot-by-quantization`의 오타는 의도된 것이 아닐 수 있으나 그대로 둔다.

---

## 7. 다음 단계 (§5 실행 계획 요약)

1. ~~7차 적대적 검토~~ **완료 + 발견 사항 전체 반영 완료** (2026-08-05,
   `review/review_findings_round7.md` + `paper/revision_provenance.md` (a)–(e))
2. 연속형 채점 파이프라인 (부분 테스트 통과율 + 토큰 로그확률) — **전제 조건**
3. 탐지기 채점 파이프라인 (CDD / perplexity / Min-k%). CDD의 문항당 다중 샘플 비용을 견적에 반영,
   2번과 기저 생성물 공유
4. LiveCodeBench `release_v6` 문항 수 확인 완료: 공통 경계 2025-01-01에서 pre 873 / post 182. 각 ≥1,000 목표 미달이므로 Q2는 보조·신뢰구간 분석
5. training cutoff 검증: Llama-3.1-8B는 LLMLagBench 리더보드에서 완료(선언 2023-12/탐지 2023-03),
   Qwen2.5는 공식 출시일을 보수적 cutoff로 사용한다. Olmo3-7B/Olmo3.1-32B 공식
   모델 카드는 모두 cutoff를 2024년 12월로 명시한다. 월 단위 보수성을 적용한
   **공통 LCB-post 시작일은 2025-01-01**. Qwen2.5 arm 경계는 2024-09-20으로 유지. **경계가 상·하한으로 나뉘는 arm은
   경계 양쪽으로 Q1b/Q2 민감도 재분석** (선언=주/탐지=민감도 사전 고정, 라벨만 변경이라 추가
   GPU 비용 0; 모호 창 탐지기 점수 비교는 Q1b 라벨로 역주입 금지 — §4.2)
6. TRACER(arXiv:2605.24079)로 잔여 오염 측정 — **Q1b의 전제 조건**. 공개 코드 없음이 확인되어
   재구현 필요(프롬프트·임계값은 원문 부록에 공개, 7차 검토 부록 A 참조). Olmo3 코퍼스 검색이
   재구현 충실도 실측 장치를 겸한다
7. 파일럿: Qwen2.5-7B와 Olmo3-7B, BNB-nf4 arm 우선. **다섯 값을 함께 측정** — (a) Q1a의 d,
   (b) Q1b의 AUC와 정밀도 간 상관 r, (c) Q2의 로그오즈 효과 크기와 문항 수준 r, (d) 조건별·
   모델별 실제 기저율, (e) Olmo3 arm의 대리 라벨 오류율 e (§5 5번의 코퍼스 검색 라벨 기준).
   CDD 관문(§4.6) 먼저 통과 확인
8. 파일럿 값으로 검정력 재계산 → 본 실험
9. **문항 단위 원자료 전부 저장** (집계값만 저장하면 대응·혼합효과 분석이 불가능해진다)
