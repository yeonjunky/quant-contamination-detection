# 7차 적대적 검토 (2026-08-05)

**대상:** `paper/paper_draft.md` (정본) + `paper/paper_draft_ko.md` (미러) + 파생 문서(`CLAUDE.md`, `pipeline_build_plan.md`)
**목적:** CLAUDE.md §7 1번 항목 — 통합 과정의 전사 오류 점검. 추가로 통계표 전수 재계산,
PDF 5편 자구 대조, 영/한 미러 대조, 파생 문서 동기화 점검.
**방법:** 모든 수치는 독립 재계산(numpy/이분탐색, 승수 2.8016 규약), 모든 인용은
`pdfs/` 원문과 자구 대조(공백 제거 검색 + 2단 조판은 좌/우 컬럼 분리 추출).

---

## 1. 검증 완료 — 이상 없음

### 1.1 통계표 (전수 재계산 일치)

| 항목 | 결과 |
|---|---|
| 승수 z_{α/2}+z_{power} = 2.8016 | ✓ |
| Q1a 필요 문항 (§4.5.1): d=0.3→87.2, d=0.2→196.2 | ✓ (표기 87/196) |
| Q2 비페어링 p=0.5 (§4.5.3): 20pp→196.2, 10pp→784.9, 5pp→3139.6 | ✓ (표기 196/785/3,140) |
| Q2 검정력 표 6×3 (§4.5.3) | 18셀 중 17셀 ±0.01 내 일치 (예외 1건 → §2.2) |
| Q2 검출 하한 15.5pp @ n=164, 청정 무한 | ✓ (재계산 15.47) |
| Q1b SE(AUC) 사다리 (§4.5.2, AUC=0.70): 0.0287/0.0212/0.0158/0.0116 | ✓ |
| Q1b 검출 한계 12셀 (r=0/0.8/0.9 × n=164/300/542/1000) | ✓ 전부 일치 |
| "정확히 풀면 170" (0.05 목표, r=0.8) | ✓ (경계 관례 차이로 170/171) |
| 라벨 잡음 표 (SE도 감쇠 방식): 170/287/541/1268 | ✓ (±1은 경계 관례) |
| CDD 관문 표 (§4.6) ΔAUC/한계/판정 5행 | ✓ (d′-스케일 10% 감쇠 모형으로 정확 재현) |
| 관문 손익분기 0.7936 | ✓ (재계산 0.7936) |
| §4.5.3 가짜 교차효과 표 (σ=1.5, 절편 보정 주변화): −1.4/−2.2/−2.5/−2.2pp | ✓ 소수점까지 재현 |
| 잔여 편향 "최대 ~0.023" | ✓ (재계산 최대 −0.0227 @ β=0.75) |
| "target의 ~11–12%" (0.023/0.2007) | ✓ (11.5%) |
| "%p 대비 비례적으로 약 4× 작음" | ✓ (50%/11.5% ≈ 4.3) |
| 비페어링 실제 기저율 557 (§4.5.3 분해 표 2행) | ✓ — 귀무 분산 규약(기저율 셀만 사용, 2×0.1275+2×0.2275=0.71 → 0.71×784.9=557.3)으로 정확 재현. 단 §2.3 참조 |
| 초록 "17×–1,000×" | ✓ (7B/410M=17.1, 70B/70M=1000) |

### 1.2 PDF 자구 대조 (pdfs/ 5편)

**arXiv:2603.03203** — 다음 전부 원문 자구 확인:
- 제목 완전 일치. CDD 전개("Contamination Detection via output Distribution") 및 Dong et al. 2024 귀속 ✓
- *"The gap is largest precisely where it matters most…"* — **§4.2 Results 절 위치 확인** (추출 텍스트 위치: "4.2" 헤더와 "5 Discussion" 사이). CLAUDE.md §4.2의 기록과 일치
- 초록 동어반복 문장, 결론의 *"including those where CDD fails entirely"* ✓
- *"a memorization threshold governs detectability"*, *"transitions sharply from chance to >90%"*, *"interaction of model size, adapter rank, and training duration"* ✓
- *"LoRA r=8 on a 7B model yields roughly 4M trainable parameters; the same rank on our 70M model yields only 98K…"* — **§5 Discussion 본문 자구 확인** (문장이 페이지 경계에 걸려 있어 단순 검색은 실패함; 이는 추출 아티팩트, §3.1 참조)
- *"the relevant factor is not the LoRA rank itself but the absolute number of trainable parameters"* ✓
- *"should not be extrapolated to larger scales without further investigation"* — §7 Limitations 위치 ✓
- *"gives CDD every advantage"* (Youden 재선택 문맥), ξ=0.01 고정(7B 교정), n=50 샘플링 ✓
- Table 1의 3.1M/9.4M/25.2M ✓ — 본문 §2.4의 "3–25M은 재현 논문 자신의 r=256 구성" 서술과 정합

**arXiv:2410.16454** — *"retains an average of 21% … increases to 83% after 4-bit quantization"* 자구 확인, ICLR 2025 게재 표기 확인. 단 §2.6 참조 (한정어 누락).

**arXiv:2605.15138** — 제목 "Forgetting That Sticks: Quantization-Permanent Unlearning via Circuit Attribution" PDF 메타데이터로 확인(본문 텍스트 레이어에는 제목 미포함). 초록의 *"per-parameter updates lie 47–828× below the NF4 quantization bin width"* 및 "sparsity-permanence tradeoff" 자구 확인.

**arXiv:2310.10628** — 제목 "Data Contamination Through the Lens of Time" 완전 일치.

**arXiv:2511.12116** — LLMLagBench 확인. 단 §2.5 참조 (제목 절단).

### 1.3 영/한 미러

- 섹션 헤더 35=35, 구조 완전 일치
- 핵심 수치 27종 grep 개수 전부 일치 (0.7936, 785, 549, 557, 170/287/541/1268, 15.5, 5.6pp/7.7pp/−2.2pp, 0.023, 47–828, 21%/83%, 3,140, 164/378/542/1,000 등)
- verbatim 인용 7종 전부 한국어판에 영어 원문 그대로 보존 ✓

### 1.4 상호참조

- 본문 인용 arXiv ID 30종 전부 References 수록 ✓ (역방향 예외 1건 → §2.4)
- 본문 § 참조 표적 섹션 전부 실재 ✓

---

## 2. 발견 사항 (심각도순)

### 2.1 [중] §4.5.3 분해 표 3행 — 549는 σ=1.5 모형과 불일치, 5차 지적의 봉합 방향이 틀렸음

표 3행: *"Paired (item difficulty SD=1.5 model), p=0.5 | 549 (implied r≈0.30)"*.

σ=1.5 로지스틱-정규 문항 모형이 p=0.5에서 실제로 함의하는 문항 수준 φ-상관을 직접
계산(8M 표본 MC)하면 **r = 0.293**이고, 따라서 785×(1−0.293) = **≈555–557**이다.
549는 r=0.3006에 대응하는 값으로, **모형이 함의하는 값이 아니라 반올림된 r≈0.30을
거꾸로 대입한 값**이다.

이력: 4차 검토 부수 확인이 "σ=1.5 함의 r=0.291, (1−0.291)×785=556"을 적었고, 5차
검토 #130이 549(r=0.30) vs 556(r=0.291) 불일치를 지적하며 "r 표기만 통일하면 됩니다"라고
권고했다. 현재 정본은 r 표기를 "≈0.30"으로 반올림하는 쪽으로 봉합했으나 **n=549는
반올림 전 모형값이 아니라 반올림된 r에서 계산된 채로 남았다.** 행 라벨이 "SD=1.5 model"인
이상 n은 모형값(≈555–557)이어야 하고, 549를 유지하려면 라벨을 "assumed r=0.30"으로
바꿔야 한다. 전자 권장 — 후속 본문 문장("the model's implied r≈0.29–0.30")과도 그쪽이
정합적이다.

**파급:** 계획 기준값은 785이므로 설계 결론에는 영향 없음. 그러나 CLAUDE.md §3.3이
경고하는 바로 그 유형(같은 표 안에서 서로 다른 가정으로 계산된 수치 혼재)이다.
영/한 양쪽 수정 필요.

### 2.2 [저] §4.5.3 검정력 표 n=50, 5pp 셀 (0.07) 재현 불가

정규 근사 단측 초과 확률로 0.054, 양측 양꼬리 합산으로 0.064 — 어느 관례로도 0.07이
나오지 않는다 (다른 17셀은 ±0.01 내 재현). n=50 행은 계획에 쓰이지 않으므로 실질 영향은
없으나, 표 전체를 스크립트 하나로 재생성해 모든 셀을 같은 계산 경로로 통일할 것을 권장
(CLAUDE.md §3.3 규율). 재계산 시 이 셀은 0.05 또는 0.06이 될 것으로 예상.

### 2.3 [저] §4.5.3 분해 표 — 557(2행)과 ≈555–557(3행 교정치)의 우연한 충돌

2행의 557(비페어링·실제 기저율, 귀무 분산 규약)과 §2.1 교정 후 3행 값(≈555–557,
페어링·p=0.5)은 **서로 다른 경로에서 나온 거의 같은 숫자**다. 교정 시 두 행이 같은 값으로
보이게 되므로, 표 캡션에 각 행의 계산 규약(2행: 기저율 셀의 귀무 분산; 3행: σ=1.5 함의
상관의 페어링 감쇠)을 명시해 "같은 숫자 = 같은 계산"으로 오독되지 않게 할 것.
785 vs ≈417을 합성하지 말라는 기존 각주와 같은 계열의 방어다.

### 2.4 [저] References에 미인용 항목 1건 — arXiv:2504.14655

*LeetCodeDataset* 이 References "Temporal-split contamination measurement" 절에 있으나
본문 어디에서도 인용되지 않는다. §2.2 또는 §4.2에서 인용하거나(LCB 외 시간 분할 대안으로
언급할 자리가 실제로 있음) 목록에서 제거할 것.

### 2.5 [저] arXiv:2511.12116 제목 절단 — 즉시 수정 가능

References 표기: *"LLMLagBench: Identifying Temporal Training Boundaries"*.
PDF 원문 제목: *"LLMLagBench: Identifying Temporal Training Boundaries in **Large
Language Models**"*. 뒷부분이 절단되어 있다. PDF가 저장소에 있으므로 즉시 수정 가능.

### 2.6 [저·정밀도] 21%→83% 인용의 한정어 누락 — "utility constraints"

원문(초록·본문 동일): *"for unlearning methods **with utility constraints**, the
unlearned model retains an average of 21% … increases to 83% after 4-bit quantization."*
원문은 무제약 방법의 수치가 "misleading"이라고 직접 경고한다. 정본 초록·§1의 인용에는
이 한정어가 없다. 수치 자체는 정확하므로 오인용은 아니나, "utility-constrained
unlearning 방법 평균"임을 명시하면 원문의 자기 한정을 보존한다. 권장 수정:
"one study reports … rising from 21% to 83% **for utility-constrained unlearning
methods** (arXiv:2410.16454)".

### 2.7 [저·즉시 가능] arXiv:2410.16454 "title not independently verified" 캐비앳이 낡음

References와 CLAUDE.md §4.3이 이 논문을 제목 미검증으로 표기하나, PDF가 `pdfs/`에
있고 제목이 확인된다: **"Catastrophic Failure of LLM Unlearning via Quantization"**
(ICLR 2025). 캐비앗을 제거하고 정식 제목을 기입할 것. 같은 캐비앳 그룹 중
2505.20276·2311.04850·2409.09927은 PDF가 저장소에 없으므로 캐비앳 유지가 맞다.

### 2.8 [중·파생 문서] CLAUDE.md §7과 pipeline_build_plan.md가 정본 §4.7과 어긋남

정본 §4.7: 파일럿에서 **다섯** 값 (a)–(e) 측정, 파일럿 모델 **Qwen2.5-7B + Olmo3-7B**.
- CLAUDE.md §7 7단계: "**네 값**을 함께 측정 — (a)…(d)", 파일럿 모델 Qwen2.5-7B만 언급.
  (e) = Olmo3 코퍼스 검색 기반 실측 오류율 e — Q1b 사이징의 전제 — 가 누락.
- `pipeline_build_plan.md` 95행: "pilot_report.py — computes pilot quantities **(a)-(d)**".

정본이 우선이므로(CLAUDE.md §2) 두 파생 문서를 (a)–(e)·Qwen+Olmo3로 갱신해야 한다.
방치 시 실제 위험: 후속 세션이 CLAUDE.md만 보고 pilot_report.py를 네 값 측정으로
구현하면 e 없이 본 실험 사이징으로 진행하게 된다 — §4.7이 명시적으로 금지하는 경로
("discovering a large e after the full run is sized would invalidate the sizing").

### 2.9 [중·설계 공백] 다중비교 보정 미명시 (1–6차에서 미지적)

설계는 최소 수십 개의 검정을 낳는다: 탐지기 3종 × 정밀도 대비 3쌍 × Q1a/Q1b ×
모델 5종(+조건별). 정본 전체에 다중비교 처리(Bonferroni/Holm/FDR, 또는 계층적
사전 지정)가 한 줄도 없다. registered-report 스타일을 표방하는 설계에서 이는 공백이다.
권장: §4.5에 한 문단 추가 — (i) **확증적(confirmatory) 검정을 사전 지정·소수화**
(예: 주 검정 = "BNB-nf4 vs fp16에서 탐지기 계열(확률 기반 대표 1종 vs CDD) 간 Q1a 효과
차이" 하나 + Q1b 순위 역전 여부 하나), (ii) 나머지 전부를 탐색적으로 선언하고 CI 중심
보고, (iii) 확증 검정군 내부만 Holm 보정. 이는 문항 수 요구를 바꾸지 않으면서
(주 검정이 소수라면) p-해킹 표면적을 제거한다.

### 2.10 [중·실행 리스크] TRACER·LLMLagBench 공개 코드 실재 여부가 §6에 없음

`pipeline_build_plan.md` open assumption #7이 "No confirmed public code release found"를
기록하나, 정본 §6/§8은 이 리스크를 다루지 않는다. TRACER는 Q1b의 **전제 조건**(§4.5.2,
§5-5단계)이므로, 코드가 없어 방법론 서술만으로 재구현해야 하는 경우 (i) 재현 충실도
불확실성, (ii) 일정 리스크가 모두 Q1b에 직결된다. §6에 한 항목 추가 권장: "TRACER/
LLMLagBench의 공개 구현이 확인되지 않으면 방법론 기반 재구현을 사용하며, 재구현의
검증은 Olmo3 코퍼스 검색 대비 교차 확인(§5-5)으로 한정된다." — 실행 착수 전
코드 실재 여부 확인을 §5 선행 단계로 명시할 것.

### 2.11 [정보] 그림 3장이 본문 미참조 — 지위 불명

`figures/fig_cdd_gate.png`, `fig_power_corrected.png`, `fig_round4_corrections.png`가
정본 본문 어디에서도 참조되지 않는다. 내부 작업 산출물이라면 문제없으나, 투고 그림이라면
본문 참조가 필요하고, §2.1 교정(549→≈557)이 반영되면 `fig_round4_corrections.png`
패널 b와의 정합도 재확인해야 한다 (CLAUDE.md §2 파일 지도의 기존 경고와 동일).

### 2.12 [정보] 필요 문항 수 반올림 방향 불일치

같은 문서 안에서 785(784.9의 올림), 87(87.2의 내림), 196(196.2의 내림)이 공존한다.
"필요 문항 수"는 관례상 올림이 맞으므로 87→88, 196→197이 pedantic하게는 옳다.
단, CLAUDE.md §4가 87/196을 검증 완료 값으로 고정하고 있고 차이가 결론에 영향을 주지
않으므로 **수정을 요구하지 않는다** — 다음 개정에서 반올림 규약을 하나로 명시하는 것으로
충분 (예: "모든 n은 올림, 표기는 근사").

---

## 3. 기각한 후보 지적 (검증 결과 문제없음 — 재발 방지용 기록)

### 3.1 "4M 인용문이 원문에 없다" — 기각 (추출 아티팩트)

단순 공백 제거 검색에서 *"LoRA r=8 on a 7B model yields roughly 4M trainable
parameters"*가 NOT FOUND로 나왔다. 원인: 문장이 페이지 경계에 걸치며 사이에 부록
GSM8K 예시 텍스트가 끼어듦. 컨텍스트 검색으로 §5 Discussion 본문에 자구 그대로
존재함을 확인. **CLAUDE.md §4.2의 기록이 옳다.** (2단 조판 PDF는 좌/우 컬럼 분리
추출 필수 — 통짜 추출은 컬럼이 줄 단위로 뒤섞여 장문 인용 검색이 조용히 실패한다.
향후 대조 작업 시 주의.)

### 3.2 "'relevant factor' 인용이 한국어판에서 누락" — 기각 (줄바꿈)

grep 단일행 검색이 놓쳤을 뿐, ko 182–183행에 영어 원문 그대로 보존되어 있다.

### 3.3 "분해 표 2행 557이 재현 안 됨(573)" — 기각 (분산 규약 차이)

대안가설(하락 반영 셀 확률: 0.85/0.75/0.35/0.25)의 분산으로는 573이 나오나, 귀무
분산 규약(기저율 셀만: 0.85/0.85/0.35/0.35)으로 557.3이 정확히 재현된다. 두 관례
모두 표준적이며 표의 서술("extreme base rates shrink binomial variance: −29%",
0.71/1.00 = 0.709)과 귀무 규약이 정합. 오류 아님 — 단 §2.3의 캡션 명시 권고로 흡수.

### 3.4 "Q1a 87은 87.2의 잘못된 내림" — 지적 유보

계산상 사실이나 CLAUDE.md §4의 검증 완료 값과 충돌하고 결론 무영향. §2.12(정보)로
강등하여 기록만 남김.

---

## 4. 권고 우선순위

| 순위 | 항목 | 규모 |
|---|---|---|
| 1 | §2.8 파생 문서 동기화 (CLAUDE.md §7, build plan의 (a)–(d)→(a)–(e)) | 소 — 실행 사고 예방 |
| 2 | §2.1 분해 표 549 교정 (+§2.3 캡션, 그림 정합 §2.11) | 소 — 영/한 동시 |
| 3 | §2.9 다중비교 사전 지정 문단 신설 | 중 — §4.5 한 문단 |
| 4 | §2.10 TRACER/LLMLagBench 코드 리스크를 §6에 반영 + §5 선행 확인 단계화 | 소 |
| 5 | §2.5·§2.7 제목 2건 즉시 수정, §2.6 한정어 추가, §2.4 미인용 정리 | 소 |
| 6 | §2.2 검정력 표 재생성(스크립트 단일화) | 소 |

수정 반영 시 CLAUDE.md §3.4 규율 적용: 정본·미러 동시 수정 후 grep 대조, 그리고
이 문서가 아닌 `paper/revision_provenance.md`에 개정 이력 기록 (논문 본문에 자기 정정
서사 금지, §3.5).

---

## 부록 A. TRACER 원문 확보 후 추가 확인 (2026-08-05, 같은 날 추가)

7차 본검토 이후 arXiv:2605.24079 원문 PDF를 확보해 `pdfs/2605.24079.pdf`로 저장하고
(이로써 `pdfs/`는 6편) 전문을 검토했다. §2.10과 관련된 갱신 및 신규 발견:

### A.1 [저] arXiv:2605.24079 제목 부정확 — References 수정 필요

References 표기: *"TRACER: Semantic-Aware Fine-Grained Code Contamination Detection"*.
PDF 원문 제목: *"TRACER: **A** Semantic-Aware **Framework for** Fine-Grained
Contamination Detection **in Code LLMs**"*. §2.5·§2.7과 같은 계열의 제목 전사 오류.
영/한 References 모두 수정할 것.

### A.2 [갱신] §2.10의 "코드 실재 미확인" → "코드 부재 확인"으로 격상

PDF 전문에서 자체 코드/데이터 저장소 링크를 찾지 못했다 (참고문헌의 타 프로젝트
GitHub 링크만 존재). build plan open assumption #7의 TRACER 부분은 가정이 아니라
**확인된 사실**이다: TRACER는 방법론 서술 기반 재구현이 필요하다. 재구현 가능성 평가:

- **재구현 난이도 낮음.** 4단계 파이프라인(정규화 → 임베딩 선별 → LLM 정밀 판정 →
  자명 과제 필터)의 프롬프트 전문이 부록 Table 7/8/9에 공개, 임베딩 모델
  (jina-embeddings-v3, 오픈)과 선별 임계값(τ_low=0.6, τ_high=0.9) 명시, LLM 백본
  교체 가능(GPT-5/Gemini-2.5-Pro/gpt-oss-120b/Qwen3-14B로 검증됨; fine-grained F1
  각각 0.86–0.99 / 0.84–0.92 / 0.63–0.90 / 0.66–0.86 범위).
- **비용 기준점.** 원논문 Table 6: 임베딩 선별이 LLM 토큰 92.1% 절감
  (163.59M→12.86M, GPT-5 비용 $1,635.9→$128.6, ~200만 쌍 세팅).
- **재구현 충실도는 설계가 이미 실측 가능.** §5 5단계의 Olmo3 코퍼스 검색 ground-truth
  대비 TRACER 채점이, 재구현 상황에서는 그대로 재구현 충실도 측정으로 기능한다.
  "공개 코드 부재" 리스크는 측정 불가능한 불확실성이 아니라 측정되는 수치가 된다.
  §2.10의 §6 반영 권고 문구는 이 취지로 쓰면 된다.

### A.3 [중·설계 문구] §5 5단계가 TRACER의 대상 코퍼스를 명시하지 않음

TRACER의 정의는 f: **D_train** × D_test → Y — 비교 대상 학습 코퍼스 없이는 실행
자체가 불가능하다. 원논문의 D_train은 공개 SFT 데이터셋 3종(CodeAlpaca /
Evol-CodeAlpaca / Magicoder)이다. 본 설계에서 실행 가능한 유일한 형태는 **"Olmo3
공개 사전학습 코퍼스에 대한 TRACER"**뿐이다 (Qwen2.5·Llama-3.3 코퍼스는 비공개 —
§4.5.2가 이미 인정하는 사실과 정합). 그러나 §5 5단계의 현재 문구 *"Measure residual
contamination in LCB pre/post and HumanEval via TRACER"*는 어느 코퍼스에 대한
측정인지 말하지 않아, 모델 5종 전부에 대한 측정으로 오독될 여지가 있다. §6의
"model-local" 항목과 일관되게, 5단계 문구에 대상 코퍼스(Olmo3)를 명시할 것.

### A.4 [정보·실행] 사전학습 코퍼스 규모에는 검색 선행 단계가 필수 (TRACER)

원논문의 쌍 공간은 ~200만 쌍(SFT 규모)이나, Olmo3 사전학습 코퍼스는 수십억 문서
규모라 전수 임베딩 자체가 불가능하다. §5 5단계의 기존 문구 "search … by n-gram and
semantic match"가 사실상 이 검색(retrieval) 선행 단계다. 실행 구조는: n-gram/BM25
검색으로 문항당 후보 top-k 추출 → TRACER 4단계를 후보 쌍에만 적용. 착수 전 확인
필요: 기존 공개 인덱스(예: infini-gram)의 Olmo3 코퍼스 지원 여부 — 미지원 시 자체
인덱스 구축이 별도 엔지니어링 항목이 된다 (build plan open assumption #6의 디스크
용량 확인과 연동).

---

## 부록 B. LLMLagBench 원문 검토 및 설계 변경의 증거 기반 (2026-08-05, 같은 날 추가)

`pdfs/2511.12116.pdf` 전문 검토 결과. 이 증거가 같은 날의 모델 로스터 변경
(`paper/revision_provenance.md` 2026-08-05 항목)의 입력이 되었다.

### B.1 방법 요약과 재구현 가능성

3단계: (i) 2021–2025 뉴스 ~8만 건 → DeepSeek-V3-0324로 후보 질문 ~8,415개 추출 → **수작업
검증으로 1,713개 선별**; (ii) 표준 답변 프롬프트(전문 공개) + DeepSeek-V3 심판의 0–2점 충실도
채점(인간 채점자와 κ=0.81/0.83); (iii) 시간순 점수열에 PELT 변화점 탐지(파이썬 `ruptures`로
구현 가능). 거부율을 병행 추적 — 선언 cutoff 이후 질문을 학습된 정책으로 거부하는 모델과
실제로 모르는 모델을 구분하기 위함.

**가용성 (§7 Availability 자구 확인):** 리더보드는 공개
(huggingface.co/spaces/pelcra/llmlagbench), **질문 데이터셋은 유출 방지를 위해 의도적 비공개**
(*"The underlying dataset of time-sensitive questions is withheld to prevent leaks"*), 평가는
이메일 요청(pelcra@uni.lodz.pl). → TRACER(코드 부재, 재구현 필수·용이)와 정반대 프로파일:
방법은 공개, 데이터가 장벽. 재구현은 가능하나 수작업 질문 검증이 병목이며, 우리 용도(선언
cutoff 주변 창만 조밀하게)로는 300–500문항 규모로 축소 가능 — 최후 수단으로만.

### B.2 결과 표에서 확인된 설계 관련 사실

| 모델 | 선언 cutoff | 탐지 변화점 | 비고 |
|---|---|---|---|
| meta-llama/Llama-3.3-70B-Instruct | 2023-12 | 2023-02 / 2023-09 | 탐지가 선언보다 이름 — "선언 이후 지식" 신호 없음 (오염 경계로는 안심 방향) |
| meta-llama/Llama-3.1-8B-Instruct | 2023-12 | 2023-03 | 동일 패턴. 채택된 신규 arm |
| meta-llama/Meta-Llama-3.1-70B-Instruct | 2023-12 | 2023-02 / 2023-09 | 2505.20276 사전 정보의 측정 대상 모델 |
| Qwen/Qwen2.5-Omni-7B | 미선언 | **탐지 실패** | 충실도 0.04, **거부율 87%**, 변화점 0건 — 방법의 실패 모드 실증 |

**정정 기록:** 본 검토 세션 초기에 Qwen2.5-Omni-7B를 "탐지 2023.12"로 잘못 읽었다(2단 표
추출 깨짐). 전체 행 복원 결과 2023.12는 모델 카드의 지식 cutoff 컬럼이고 탐지는 0건이다.
이 정정 자체가 "리더보드 등재 ≠ 사용 가능한 검증"의 실례로 설계 논의에 반영되었다.

### B.3 리더보드 ~35모델 대비 설계 적합성 필터

dense(비-MoE) · 텍스트 전용 · 가중치 공개 · 단일 GPU 적재의 4중 필터를 통과하는 모델은
**Llama-3.1-8B-Instruct가 사실상 유일**하다. 탈락 사유 대표 예: Llama-4-Scout/Mixtral/
gpt-oss/kimi-k2(MoE), gemma-3 계열(멀티모달+QAT), gpt-oss(native MXFP4 학습 — fp16 기준선
부재로 PTQ 사다리 성립 불가), c4ai-command-r-plus(cutoff ≈2023-01이 LCB 시작(2023-05)보다
일러 LCB-pre 조건 소멸), 나머지는 비공개 가중치 또는 400B+.

### B.4 파생 확인 사항

- **정정 (기록 오류):** 이 부록의 최초 판본은 References 제목 수정(2511.12116 절단 복원,
  2410.16454 캐비앗 해제)이 "같은 날 수정 완료"라고 적었으나, 당시 실제로는 CLAUDE.md만
  갱신되었고 정본 References는 미수정 상태였다 — CLAUDE.md §3.4가 경고하는 "고쳤다고 표시하고
  안 고침" 패턴의 자기 사례. 잔여 항목 반영 단계(provenance (e))에서 grep으로 재확인 후 정본
  영/한 References 3건(2511.12116 전체 제목, 2410.16454 정식 제목, 2605.24079 정식 제목)이
  실제로 수정되었다.
- §5 4단계의 실행 경로가 모델별로 명시됨: Llama-3.1-8B = 리더보드 인용, Qwen2.5 = 평가 요청,
  Olmo3 = 코퍼스 날짜 메타데이터(프로빙보다 강한 증거).
