# 8차 적대적 검토 (2026-08-05)

**대상:** `paper/paper_draft.md` (정본) + `paper/paper_draft_ko.md` (미러)
**목적:** 정본 전문에 대한 비판적 재검토 — 통계표 전수 재계산, PDF 6편 자구 대조,
영/한 미러 수치 대조, 내부 일관성 점검.
**방법:** 모든 수치는 독립 재계산 (numpy + 수치 적분/이분탐색, scipy 없음, 승수 2.8016 규약).
모든 인용은 `pdfs/` 원문과 자구 대조 (공백·구두점 제거 정규화 검색 + 2단 조판은 좌/우 컬럼
분리 추출 + 전폭 추출 병행). 영/한 대조는 핵심 수치 25종의 출현 횟수 grep 대조.

**결론 요약:** 인용은 전부 원문과 일치, 재계산한 통계표는 전부 재현됨. **확정 오류 2건**
(둘 다 영/한 양쪽에 존재, 7차 검토와 provenance 어디에도 없는 신규 발견) + 정밀도/일관성
이슈 5건 + 경미 1건. CLAUDE.md §3.2 규율대로 **반영 전 독립 재검증 필요.**

---

## 1. 확정 오류 (수정 필요)

### 1.1 §3.0이 §4.5.2와 모순 — n=164에서의 Q1b 검정력

- 위치: `paper_draft.md:269` / `paper_draft_ko.md:265–266`
- §3.0: *"Q1b can detect a 0.05 AUC difference at n=164 with paired detector scores (§4.5.2)"*
- §4.5.2 자신: n=164의 검출 한계는 **0.051**로 목표 0.05에 *"just short"* — 정확히 풀면 **170문항**.
- 독립 재계산: n=164, AUC=0.70, r=0.8에서 검출 가능 ΔAUC = 2.8016 × √(2(1−0.8)) ×
  SE_HM(0.70, 164) = 2.8016 × √0.4 × 0.0287 = **0.0509**. §4.5.2가 맞고 §3.0이 틀렸다.
- **수정안:** §3.0을 "≈0.05(0.051)" 또는 "170문항이면 0.05" 형태로. 초록의 *"well powered
  even at benchmark-imposed sample-size ceilings (e.g., 164 items)"*도 d=0.2에서는
  196(홀름 보정 시 279) > 164라 과장이지만, 요약 화법 범위로 판단 — 완화 권고에 그침.

### 1.2 §4.5.3 "r=0.6–0.9 → 157–314" — 범위와 숫자 불일치

- 위치: `paper_draft.md:587` / `paper_draft_ko.md:573`
- 문장: *"could plausibly reach 0.6–0.9, which would bring the requirement down to 157–314"*
- 초안 자신의 공식 n = 785×(1−r)로: r=0.6 → **314**, r=0.8 → **157**, r=0.9 → **79**.
- 즉 157–314는 r ∈ [0.6, **0.8**]에 해당. r=0.9 끝점은 79이지 157이 아니다.
- **수정안:** 범위를 "0.6–0.8"로 고치거나 숫자를 "≈79–314"로 고치거나, 둘 중 하나.
  (같은 표의 §4.5.3 분해 행들이 정확히 이 공식으로 계산되어 있으므로 공식 자체는 정본과 일치.)

---

## 2. 정밀도 / 일관성 이슈 (권고)

### 2.1 §4.6 관문 표 — "10% reduction in separation"의 계산 척도 미기재

- 위치: `paper_draft.md:676–682` (ko `:653` 부근)
- ΔAUC 열(0.002/0.010/0.019/0.026/0.019)은 0.1×(AUC−0.5)가 **아니다** (그렇게 읽으면
  뒤 세 행이 0.020/0.035/0.045로 어긋남). 재계산으로 확인한 실제 계산:
  **ΔAUC = AUC − Φ(0.9·Φ⁻¹(AUC))** — 프로빗 척도 분리도(d′)의 10% 감소.
- 수치 자체는 전부 맞음 (7차 검토 §1.1도 d′-스케일로 재현 확인). 문제는 **캡션에 계산
  경로가 없다**는 것 — CLAUDE.md §3.3 자체 규율 위반. 캡션 한 구절로 해결.

### 2.2 fp16 vs bf16 기준선 표기 불일치

- §4.1 표·본문(`:331–357`)은 "fp16", §4.3(`:445`)은 "fp16 / bf16 (baseline)", §4.6은 "16-bit".
- Qwen2.5·Llama-3.1·Olmo3 체크포인트는 전부 **bf16 배포**이고 fp16↔bf16은 numerics가
  다르다 — 정밀도 numerics를 다루는 논문에서 흐릴 수 없는 구분. Q1a의 기준선 고정
  대응 비교가 이 baseline에 정의되므로 실측 dtype(bf16)으로 통일 권고.

### 2.3 "HumanEval, MBPP+, both released 2021" (`:417`)

- MBPP는 2021, **MBPP+**(EvalPlus 테스트 증강판)는 2023 공개. 오염 관점에서 유효한 날짜는
  문제 지문의 2021이므로 논증은 유지되나 문장은 부정확.
- **수정안:** "문제 지문은 2021년(MBPP) 기원; plus 변형의 테스트 스위트는 이후 추가" 취지로.

### 2.4 LCB pre-cutoff ≥1,000 목표 — Llama-3.1-8B arm에서 산술적으로 불가능, 위험 논의는 post 쪽만

- §4.2 표(`:377–380`)의 pre-cutoff "Target ≥ 1,000"에 arm별 단서 없음. 그러나 의심(pre) 풀은
  arm별로 해당 arm의 cutoff에 묶임: Llama-3.1-8B의 의심 창은 2023-05(LCB 수집 시작, 초안
  `:423`) ~ 2023-12 — 약 7개월치 수집분(수백 문항 규모), 탐지 경계 민감도 런에서는 **0**
  (초안 `:424`가 스스로 인정).
- 풀 축소 위험 논의(`:387–390`)는 post-cutoff 쪽만 다룸. 또한 두 ≥1,000 목표를 동시에
  채우려면 2024-09 이후 경계 양쪽으로 LCB 문항 총 ≥2,000이 필요 — 이 산술을 초안 어디서도
  대면하지 않음. §5 3단계가 실측할 일이지만, pre 쪽 위험을 post와 대칭으로 명시할 것.
  (LCB 실제 보유량은 본 검토에서 실측하지 않음 — 오류가 아니라 위험 플래그.)

### 2.5 참고문헌 제목 — CSV 축약형 상속, 주석은 3건뿐 (`:890–940`)

- `reference/contamination_literature.csv`의 제목 축약 경고(CLAUDE.md §2)에도 불구하고
  초안 참고문헌이 CSV 제목을 자구 그대로 복사함 (grep 대조 확인: "CodeCleaner:
  Contamination Mitigation Toolkit", "Overestimation in LLM Evaluation", "How Much Do
  LLMs Cheat? One-Time-Pad Framework", "LeetCodeDataset: Temporal Dataset for Robust
  Evaluation" 등).
- "투고 전 확인" 주석은 2311.04850 / 2409.09927 / 2505.20276 세 건에만 존재.
- **수정안:** PDF 검증분(pdfs/ 6편 + 7차에서 확인된 2605.24079·2511.12116) 외 전 항목에
  주석을 달거나, 지금 arXiv 전수 대조를 수행.

### 2.6 (경미) `:532` "rises to Q2's level (≈542)" — 척도 혼합형 표현

- 541(e=20%에서의 Q1b 요건) / ≈555(Q2 페어링 요건) / 542(§4.2가 "참고값일 뿐 분석 셀이
  아니다"라고 못박은 HumanEval+MBPP+ 풀)라는 **유래가 다른 세 숫자**가 우연히 근접해
  한 괄호에 뭉개짐. CLAUDE.md §3.3이 경고하는 패턴.
- **수정안:** "rises to ≈541 — the same order as Q2's paired-model requirement (≈555)".

---

## 3. 검증 완료 — 이상 없음

### 3.1 통계 전수 재계산 (전부 일치)

| 항목 | 결과 |
|---|---|
| Q2 검정력 격자 18셀 (§4.5.3) | ✓ 전부 일치 |
| 196 / 785 / 3,140 (20/10/5pp) | ✓ (196.2 / 784.9 / 3139.6) |
| 분해 표: 557 (귀무 기저율 분산 규약) / 555 (r=0.293089 수치 적분) | ✓ 정확 재현 |
| 가짜 교차효과 표 (σ=1.5 주변화): 2.6/4.0/−1.4 · 5.6/7.7/−2.2 · 8.8/11.2/−2.5 · 12.3/14.5/−2.2 | ✓ 소수점까지 재현 |
| 집계 로짓 잔여 편향 최대 0.0231 @ β=0.75 (문서 "~0.023") | ✓ |
| SE(AUC)=0.0287 @ n=164/클래스, AUC=0.70 (Hanley–McNeil, 양 클래스 각 n) | ✓ |
| ΔAUC 검출 한계 12셀 + 관문 표 검출 한계 5행 (n=542/클래스, r=0.8) | ✓ 전부 일치 |
| 라벨 잡음 표 170/287/541/1268 — 정본 채택 "SE도 감쇠" 열 | ✓ |
| MDE 15.47 → 표기 15.5pp | ✓ |
| 홀름 α/4 승수 3.339 → 124 / 279 | ✓ (123.9 / 278.7) |
| 관문 손익분기 ≈0.79 | ✓ (0.7936) |
| "정확히 풀면 170" | ✓ — n=170에서 0.050001, 경계선상이나 초안 값 유지가 맞음 (CLAUDE.md §4.1과 일치) |
| 초록 "17×–464×", "one to two and a half orders of magnitude" | ✓ (7B/410M=17.1, 32.5B/70M=464.3; log₁₀ 1.23–2.67) |

### 3.2 PDF 자구 대조 (pdfs/ 6편)

**arXiv:2603.03203** — 다음 전부 자구 확인: *"The gap is largest…"* / 초록 동어반복 문장 /
결론 *"including those where CDD fails entirely"* / *"should not be extrapolated…"* /
*"detectable by simpler methods"* / *"gives CDD every advantage"* / 임계값 원 논문 ξ=0.01
고정 vs Youden 재선택 / *"the relevant factor is not the LoRA rank itself but the absolute
number of trainable parameters"* / **~4M Discussion 문장** (*"LoRA r=8 on a 7B model yields
roughly 4M trainable parameters; the same rank on our 70M model yields only 98K…"*) —
p.7에서 그림 범례가 문장 중간에 끼어 추출되어 통짜 검색은 실패, 컬럼 맥락 복원으로 확인
(7차 §3.1 경고 그대로) / r=256 = 3–25M 문장 / Table 1의 3.1M/9.4M/25.2M / 샘플링 프로토콜
(greedy t=0 + **n=50** @ t=0.8, *"matching the original paper"*) / Dong et al. (2024) 귀속 /
Sela, Tel Aviv University.

**arXiv:2511.12116** — Table 3 행 자구 확인: `meta-llama/Llama-3.1-8B-Instruct 8B
2024.07.21 | 2023.12 | 2023.12 | 2023.03` (변화점 2023-03-26). Qwen2.5-Omni-7B: 변화점
**0개**, 거부율 **0.87**, *"no detectable changepoint due to consistently low performance"* ✓.

**arXiv:2605.15138** — 렌더링 제목은 추출 불가(1면 텍스트가 저자부터 시작)였으나 **PDF
메타데이터 Title로 정식 제목 확인**: *"Forgetting That Sticks: Quantization-Permanent
Unlearning via Circuit Attribution"*. 초록 자구: *"per-parameter updates lie 47–828× below
the NF4 quantization bin width"* ✓.

**arXiv:2410.16454** — *"for unlearning methods with utility constraint… an average of
21%… increases to 83% after 4-bit"* 취지 자구 확인 (utility-constraint 한정어 포함) ✓.
제목 ✓.

**arXiv:2310.10628** — 제목 자구 ✓. **arXiv:2605.24079** — TRACER 3계층(functionally
identical / nearly identical / shared logic) 자구 ✓.

### 3.3 영/한 미러 대조

핵심 수치 25종(87, 196, 785, 555, 557, 3,140, 124, 279, 3.339, 0.051, 170, 287, 541,
1,268, 15.5, 0.023, 0.293, 464, 17×, 0.79, 415, 419, 157, 314, 2.6pp) 출현 횟수 전부 일치.
→ §1의 오류 1·2는 **양쪽 판 동일 위치에 존재**하므로 수정 시 같은 턴에 양쪽 반영할 것.

---

## 4. 재현 방법 (검증용)

- 통계: numpy만 사용. Hanley–McNeil SE(양 클래스 각 n 규약), 승수 2.8016, 대응 검출 한계
  = 승수 × √(2(1−r)) × SE. σ=1.5 로지스틱-정규 주변화는 z∈[−8,8] 20만 점 사다리꼴 적분,
  기저율 0.85/0.35는 α 이분탐색으로 절편 보정. 관문 표는 ΔAUC = AUC − Φ(0.9·Φ⁻¹(AUC)).
- PDF: pdfplumber로 좌/우 컬럼 분리 + 전폭 추출 병행, 소문자·영숫자만 남긴 정규화 문자열
  에서 검색. 실패 시 원시 텍스트 정규식으로 맥락 복원 (§3.2의 ~4M 사례).
