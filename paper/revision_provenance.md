# Revision Provenance — `paper_draft.md`

이 파일은 `paper_draft.md`의 부록 A에서 분리한 개정 이력이다. 논문 본문에는 내부 검토 이력을
싣지 않기로 하여 별도 문서로 옮겼다. 감사 추적(audit trail) 용도이며 투고 원고에는 포함하지
않는다. 상세 원본은 `review_findings_round1.md`(= `review_findings.md`)~`review_findings_round5.md`
및 `review_response.md` 참조.

---

## 개정 이력 (6차 검토까지)

This design synthesizes and supersedes `contamination_literature_review.md` and
`topic3_experiment_plan.md` after five rounds of adversarial review. Notable corrections carried forward
(full detail in `review_findings_round1.md` through `review_findings_round5.md` and `review_response.md`):

- Round 1: corrected a fabricated effect-size citation (arXiv:2505.20276 misquoted as "1–4%p"; actual
  figures are 0.8%/59%, long-context only); narrowed an over-generalized claim from arXiv:2603.03203;
  corrected a self-contradictory power calculation; identified the base-rate confound (§4.5.3) and the
  HumanEval sample-size ceiling (§4.5.3) as previously undiscussed design flaws.
- Round 2: caught that narrative documents had not actually been updated despite being marked complete;
  found new errors introduced during revision (a truncated arXiv ID, a broken figure reference); diagnosed
  (then, in round 3, retracted the diagnosis of) an apparent simulation bug in the base-rate-confound
  table, which was actually two internally consistent but different statistical models.
- Round 3: confirmed round 2's retraction was correct; identified that a reported Monte Carlo "error" of
  0.023 was partly systematic bias, not noise; connected the paired-design sample size (560) to the
  base-rate-confound model's implied correlation (r≈0.29).
- Round 4: promoted Q1 (contamination-signal detection) to primary and Q2 (pass@1 interaction) to
  secondary, based on a power comparison; caught a scale-mixing error in combining base-rate and pairing
  corrections (785 × implied-r shortcuts do not compose); caught an off-by-one-item misreading of the
  164-item HumanEval power ceiling; introduced, then self-corrected within the same round, an
  extrapolation argument about arXiv:2603.03203 that violated the very non-extrapolation caveat it cited.
- Round 5: verified round 4's corrections via direct PDF text comparison against arXiv:2603.03203; raised
  four residual issues, of which **three were adopted** — the over-read "works in all conditions" claim in
  §2.4, a label-noise table entry computed with an inconsistent method (§4.5.2, e=10% row: 265 → 287, with
  1,268 stated explicitly for e=30%), and an over-broad ξ-recalibration instruction narrowed to apply only
  when CDD accuracy (not AUC) is reported (§4.4).
- Round 6 (independent re-verification against `2603.03203.pdf`): re-derived all four round-5 items from
  the source text and from first principles. Findings:
  - The label-noise arithmetic is confirmed. Holding SE fixed at AUC=0.70 gives 170 / 265 / 471 / 1,060
    for e = 0/10/20/30%; propagating the attenuation into SE as well gives 170 / 287 / 541 / 1,268. The
    round-4 table mixed the two, and §4.5.2 now uses the second (internally consistent) column throughout.
  - The ξ and quotation items are confirmed verbatim ("This gives CDD every advantage"; "The gap is
    largest precisely where it matters most…"), with the added nuance that the source's Conclusion states
    the probability-vs-CDD claim more strongly than its abstract (§2.4).
  - **Round 5's third item is rejected.** It claimed the "~4M" figure for LoRA r=8 on a 7B model was absent
    from the source and should be replaced by "3–25M." The source states the 4M figure verbatim in §5; the
    3–25M range describes the replication's *own* r=256 configurations on 70M–410M models. Adopting the
    substitution had merged two distinct quantities, and §2.4 is corrected accordingly. The round-5 error
    arose from reading Table 1 (70M/160M/410M only) as exhaustive of the paper's parameter counts.
  - Additionally: the CDD initialism *is* expanded in the source ("Contamination Detection via output
    Distribution," attributed to Dong et al. 2024), so §4.4's refusal to assert an expansion is lifted,
    and §2.4's attribution is reworded so that CDD is not implicitly credited to the replication's author.

- Round 7 (2026-08-03, external review — `review/paper_draft_review.md` and
  `review/model_recommendations.md`). The first round to raise an *executability* objection rather than a
  citation or arithmetic one. Independently verified before adoption, per the discipline established in
  round 6.

  **Adopted:**
  - **The 70B fp16 baseline is not runnable on the available hardware.** Re-derived: Llama-3.3-70B at fp16
    is 141.2 GB of weights, which exceeds an H200's 141 GB before any KV cache, and far exceeds an H100's
    80 GB. §4.1's previous instruction that this baseline "must be re-run" was therefore an impossible
    step, not a pending one. §4.1 now records the hardware constraint explicitly, plans a one-time
    multi-GPU rental for the complete fp16 pass, and pre-specifies an int8-anchored three-level ladder as
    the fallback. §6 gains a corresponding threat entry.
  - **The rental window was under-scoped in the review** and is corrected here: it must cover CDD's
    multi-sample generations *and* the teacher-forced log-probability passes perplexity and Min-k% require,
    not generations alone (§4.1, §4.4).
  - **Mechanism extrapolation, distinct from scale extrapolation.** arXiv:2603.03203 injects contamination
    via LoRA fine-tuning; this design studies contamination arising naturally in pretraining. §2.4 noted
    this in a single clause but §6 carried only the scale version. §6 now treats the two as independent
    axes and states that the §4.6 pilot gate measures their composite and cannot separate them.
  - **Q1a lacked a substantive justification.** Its only stated warrant was statistical power (§3.0). §1
    now argues the point directly: contamination certifications are issued on full-precision checkpoints
    while deployed models are quantized, so a measurable per-item detector shift means an fp16 clean bill
    of health may not transfer to the int4 model inheriting it. No new citations were introduced; the
    mechanism argument re-uses §1's existing unlearning thread.
  - **Detector-family cost asymmetry** (§4.4): CDD alone bears generation cost; perplexity and Min-k% need
    a single forward pass per item per precision. The review described the latter as "one generation,"
    which is corrected here. Consequence recorded: a throughput-driven cut can only come from CDD's sample
    count *n*, and degrades only the CDD arm.
  - **Throughput measurement promoted to a pilot deliverable** (§5, step 6), since both the rental window
    and any reduction in *n* depend on it.
  - **Olmo3-7B and Olmo3-32B added to the main analysis** (author decision). Rationale: they are the only
    arms whose contamination labels can be *measured* against a released pretraining corpus rather than
    inferred from a release-date proxy, which directly addresses the label-noise dependency that §4.5.2
    identifies as a precondition for Q1b. Architecture is still not an axis (all arms are dense
    transformers), and size remains the only model-comparison axis. **Training-corpus transparency** is
    an operational selection property that enables model-local label validation for §4.5.2, not an
    effect-modifying axis or basis for cross-model contrasts. Consequential edits: §2.4
    (arXiv:2602.12413's Olmo3-corpus finding becomes a direct
    prior on one of our own arms rather than borrowed evidence), §4.2 (the post-cutoff boundary is set by
    the latest of five cutoffs, flagged as a risk to the ≥1,000-item target), §4.5.2, §4.5.3 (the 0.85/0.35
    base rates are marked as Qwen-class illustrations requiring per-model measurement), §4.7 (Olmo3-7B
    joins the pilot; a fifth measured quantity *e* is added), §5 step 5 (corpus search, and cross-validation
    of TRACER against it), and model counts 3 → 5 throughout.
  - **A limitation the review did not raise** is added to §6: Olmo3's ground-truth labels are model-local.
    They validate the labelling *method*, not the labels of the closed-corpus arms, and a measured *e*
    transfers to the other arms only as an assumption.
  - **A dangling cross-reference** to a non-existent §3.3 (§1, both language versions) was found during
    verification and corrected to §3.0. Not raised by either review document.

  **Rejected, with reasons:**
  - **The review's justification for an int8 baseline** — that arXiv:2505.20276 reports only ~0.8% loss at
    8-bit — is not adopted. That figure is a long-context (>64K) result, and §2.7 already states it "cannot
    be transferred directly" to code generation; using it here would repeat the round-1 misattribution.
    More fundamentally, pass@1 fidelity is the wrong criterion: assuming int8 detector scores approximate
    fp16 detector scores would assume the null hypothesis of the paper's own primary question, and no
    evidence exists on int8's effect on CDD peakedness, perplexity, or Min-k%. The int8 fallback is
    therefore recorded as a stated limitation, not as a justified equivalence.
  - **The review attributed the requirement for a uniform four-level ladder to §4.5.3.** §4.5.3 concerns
    the base-rate confound *between conditions*, not ladder uniformity *across models*. The actual basis is
    §4.1's within-model baseline rule; the argument was re-grounded accordingly.
  - **Qwen2.5-Coder** (model recommendation 2): would introduce code-specialization as a second
    cross-model axis, and its near-ceiling HumanEval pass@1 would worsen the base-rate and ceiling problems
    §4.5.3 already documents. Left as future work.
  - **StarCoder2** (model recommendation 3): its LiveCodeBench-post accuracy would sit near the floor,
    recreating §4.5.3's base-rate confound from the opposite end. Moot after the Olmo3 decision.
  - **Rewording "registered-report-style":** the abstract's parenthetical and §8 already disclose the
    absence of external registration twice. No change made.
  - **Lifting the "title not independently verified" caveats** on arXiv:2410.16454, 2505.20276,
    2311.04850, and 2409.09927: the review verified those papers' *content claims*, not their titles. The
    caveats and the pre-submission title-check action remain.
  - The review's speculation that §4.1's int8 footnote was a trace of a previously failed run, and its
    account of retracting its own arXiv:2404.00699 finding, are review-document material and are not
    carried into the paper.

  No table figures changed in this round; `figures/fig_power_corrected.png`, `fig_cdd_gate.png`, and
  `fig_round4_corrections.png` were left unchanged and were recorded as valid **as of that round**.

  **2026-09-18 정정.** 위의 "valid" 판정은 이후 개정으로 더 이상 성립하지 않는다. 9차 검토 D-S9에
  따라 세 그림을 다시 열어 본문과 대조한 결과, 세 그림 모두 현재 본문과 어긋난다.
  `fig_power_corrected.png`는 패널 b가 §4.2가 "never a pooled analysis cell"로 못박은
  HumanEval+MBPP+ 542문항 합동 셀에서 8.5%p를 그리고, §4.5.3의 주 Q2 기준값 14.7/16.1%p가 없다
  (패널 a의 검정력 곡선은 §4.5.3 표와 일치한다). `fig_round4_corrections.png`는 패널 a가 §4.5.3
  분해 표의 555·≈412를 549·419로 적고 옛 "785×(1−0.25)=589" 주석을 남겨 두었으며, 패널 b의
  "Q2와 동급 (542)"가 §4.5.2의 ≈555와 다르다. `fig_cdd_gate.png`는 기저 AUC 0.79 진입 관문과
  x축의 "파일럿 측정 대상" 표기가 §4.6–4.7에서 폐기한 설계이고, "Q1b 불가 구간"이라는 구성은
  §2.4의 "CDD's chance-level AUC alone does not make all of Q1b undetectable"과 반대다.
  세 그림은 영·한 두 판 어디에서도 참조되지 않는다(검색 0건). 파일별 상세와 재작성 시 고칠
  점은 **`figures/README.md`**에 기록했다. 그림 파일 자체는 이번에 바꾸지 않았다 — 생성
  스크립트가 저장소에 없고 matplotlib도 설치돼 있지 않아 이 저장소에서는 재생성할 수 없다.

---

## 2026-08-05 — 내부 적대적 검토(`review/review_findings_round7.md`) 및 설계 변경

두 부분으로 구성된다: (a) 전사 오류·인용 전수 재검증, (b) 사용자 결정에 따른 모델 로스터 변경.

### (a) 검토 결과 중 이 날짜에 정본에 반영된 것

- **arXiv:2410.16454 정식 제목 확정** — 로컬 PDF에서 확인: *"Catastrophic Failure of LLM
  Unlearning via Quantization"* (ICLR 2025). "title not independently verified" 캐비앗은 이
  논문에 한해 해제. (2505.20276, 2311.04850, 2409.09927의 캐비앗은 유지 — PDF 미보유.)
- 나머지 발견(§4.5.3 분해 표의 549 vs 모형 함의값 ≈555–557, 검정력 표 n=50 셀, 다중비교
  보정 부재, TRACER 코드 부재의 §6 반영, 21%→83% 인용의 utility-constraints 한정어 등)은
  검토 파일에 기록되어 있으며 **아직 정본 미반영** — 반영 시 이 파일에 항목별로 추가할 것.

### (b) 설계 변경: 모델 로스터 (사용자 결정, 2026-08-05)

**변경 내용:**
- **Llama-3.3-70B 완전 제거** (이전 계획: 1회성 멀티GPU 대여로 fp16 pass 확보, 실패 시
  int8 앵커 폴백). §4.1의 "Compute constraint and the 70B baseline" 절 전체와 §6의
  "Compute-constrained baseline asymmetry" 위협 항목이 삭제됨.
- **Llama-3.1-8B-Instruct 추가** (주 분석). 채택 근거: (i) 리더보드 필터 통과 — LLMLagBench
  공개 리더보드에서 training cutoff가 기검증된 유일한 설계 적합(dense·텍스트 전용·가중치
  공개·단일 GPU) 모델 (선언 2023-12, 지식 급락 변화점 2023-03); (ii) arXiv:2505.20276의
  BNB-nf4 취약성 사전 정보(Llama-3.1-70B 측정)와 같은 계열 — 기존의 3.1→3.3 버전 불일치
  캐비앗이 규모 불일치(70B→8B, ~9×) 캐비앗으로 대체됨; (iii) 전 arm 단일 장비 fp16 기준선
  확보로 Q1a 비교 가능성이 5모델 전체로 복원됨.
- **Gemma-4-31B-it 완전 제거** (이전 계획: 부록 QAT-vs-PTQ 비교 전용). 기존 3종 교란(공식
  QAT 체크포인트/thinking 모드/멀티모달)에 더해, 공식 QAT 체크포인트의 q4_0 포맷이 제2 추론
  스택(llama.cpp 계열)을 요구하고 그 스택 간 numerics 차이가 QAT-vs-PTQ 대비 자체를 교란한다는
  점이 결정적. QAT 비교는 future work로 명시.

**파생 수정:** 초록·§6의 스케일 주장 7B–70B → 7B–32B, 17×–1,000× → 17×–464×
(= 32.5B/70M), "1~3자릿수" → "1~2.5자릿수"; 기여 1·4와 §2.4·§4.5.1·§4.6·§8의 "32B–70B" →
"32B"; §4.1 모델 표·프로즈·컴퓨트 표 개편(전 모델 fp16 공유로 bf16 통일 문제 소멸);
§4.2에 Llama-3.1-8B의 선언/탐지 cutoff 격차(2023-12 vs 2023-03)와 LCB-pre 희석 리스크 신설;
§5 4단계에 모델별 cutoff 근거(LLMLagBench 리더보드/Qwen 출시일/Olmo3 코퍼스 메타데이터)를
명시; §5 6단계에서 70B 대여 창 결정 제거; §6에 "규모 불일치
사전 정보"와 "Llama-3.1-8B cutoff 불확실성" 위협 신설; §8 범위 한계에 32.5B 상한 명시.
영/한 동시 반영, 핵심 수치 grep 대조 완료 (7B–32B 6:6, Llama-3.1-8B 6:6, LLMLagBench 9:9 등).

**동기화된 파생 문서:** `CLAUDE.md` (§4.2 32B 문구, §4.3 인용 주의, §5 지침 7·8, §6 PDF 목록
6편, §7 실행 계획 — 파일럿 다섯 값 (a)–(e)로 정정 포함), `pipeline/src/qcd/models/registry.py`
(로스터 교체). `pipeline_build_plan.md`는 상단에 설계 변경 노트로 처리.

### (c) 후속 반영 (같은 날): cutoff 증거 등급 + 사전 지정 경계 규칙

사용자가 "cutoff가 발표된 ~32B 모델로 로스터를 제한하고 한계에 남기는" 방안을 제안. 검토
결과 **로스터 제한은 기각** — (i) 이 설계는 선언값 자체를 불신하는 것을 전제로 하며(§5 4단계,
§6; Llama-3.1-8B의 선언 2023-12 vs 탐지 2023-03이 실증), 선언은 필터 기준으로 부적합;
(ii) "≤32B·dense·텍스트·공개·선언 명확" 필터를 실제로 통과하는 대체 후보가 사실상 없음
(Llama-3.2-1B/3B는 기저율 바닥, CodeLlama는 cutoff가 LCB 시작 이전, Phi-4는 합성 데이터
중심 레시피가 날짜 대리 라벨을 구조적으로 약화 — 오염 논문에 최악의 조합); (iii) Qwen 제거
시 파일럿 워크호스·2계열 크기 축 복제·코드 특화 arm을 모두 상실.

대신 제안의 알맹이(컷오프 불확실성의 정직한 처리)를 **§4.2 사전 지정 경계 규칙**으로 흡수
(사용자 승인): arm별 증거 등급(코퍼스 메타데이터 > 검증된 선언 > 선언만 > 출시일 상한) 표를
§4.2에 신설하고, LCB-post 경계 = arm별 보수적 상한의 최댓값(잠정: Qwen2.5 출시일 2024-09 vs
Olmo3 코퍼스 종료 중 늦은 쪽)으로 사전 지정. 출시일은 무조건 성립하는 상한이므로 선언 없는
arm에도 방어 가능한 경계가 존재한다.
파생 수정: §5 3단계(최악 경계 기준 집계), §5 4단계(폴백 문구), §6(증거 이질성 위협 확장),
§8(arm 의존적 정밀도 한계 신설). 영/한 동시 반영, 핵심 어구 grep 대조 완료. CLAUDE.md §7
5단계 동기화.

### (d) 후속 반영 (같은 날): 경계 민감도 분석 사전 지정

사용자 제안("Llama를 선언 cutoff와 추정 cutoff로 두 번 테스트")을 채택하되 일반 규칙으로
확장해 반영. §4.2에 신설: 코퍼스 등급 미만의 모든 arm은 라벨 의존 분석(Q1b/Q2)을 상·하한
경계 양쪽으로 재실행 — Llama-3.1-8B는 선언 2023-12(주) vs 탐지 2023-03(민감도).
Qwen2.5는 출시일 상한만 사용한다. 역할(주/민감도)은 사전 고정하고 결과를 본 뒤 재조정하지 않는다.
탐지 경계 하에서는 LCB 수집 시작(2023-05)이 탐지 컷오프(2023-03)보다 늦어 Llama의 LCB-pre가
공집합이 됨을 명시 — 두 런은 실질적으로 "7개월 창을 의심 처리 vs arm을 LCB 대비에서 제외"의
비교다. 추가 비용 없음(생성·채점 불변, 분석 시점 라벨만 변경 — §5 8번의 문항 단위 원자료
요건이 근거). 모호 창 문항의 탐지기 점수 분포 비교는 기술적 점검으로만 보고하고 Q1b 라벨로
역주입 금지(순환성 가드) 명문화. 파생 수정: §5 9번(분석 단계에 민감도 항목), §6(Llama cutoff
위협을 "표기"에서 "정량화"로 격상 + 감쇠 방향의 무해성 명시). 영/한 동시 반영.

### (e) 후속 반영 (같은 날): 7차 검토 잔여 항목 전체 반영

검토 파일의 권고 우선순위 표 기준, 미반영으로 남아 있던 항목 전부를 정본 영/한에 반영:

- **§2.1 (분해 표 549):** σ=1.5 모형의 함의 상관을 수치 적분(사다리꼴, MC 아님)으로 확정 —
  **r = 0.293089, 785×(1−r) = 554.9 → 555** (4차의 556, 7차 MC의 555–557은 모두 MC 잡음).
  3행을 "555 (implied r = 0.293), −29%"로 교체, 본문 프로즈 "r≈0.29–0.30" → "r ≈ 0.293".
- **§2.3 (557 vs 555 우연 충돌):** 표 캡션 신설 — 2행은 귀무 기저율 분산 규약, 3행은 모형 함의
  상관 규약, 두 값의 근접은 우연이며 합성 불가(4행은 별도 결합 계산) 명시.
- **§2.2 (검정력 표):** 단일 공식 power = Φ(δ/SE−z)+Φ(−δ/SE−z), SE=√(4pq/n)로 전 셀 재생성.
  변경 셀 4개: 50/5pp 0.07→0.06, 164/10pp 0.24→0.25, 164/20pp 0.72→0.73, 1600/5pp 0.51→0.52.
  공식을 캡션에 명시. `figures/fig_power_corrected.png`는 이미 이 공식 기반 곡선이라 표가
  그림에 정합해진 것 — 그림 재생성 불필요 확인.
- **§2.9 (다중비교):** §4.5.6 신설 — 확증 검정군 4개 사전 지정(C1–C3: fp16→nf4 Q1a 탐지기별
  이동 @LCB, C4: Q1b 계열 순위 역전), 군내 Holm, 그 외 전부(다른 양자화 수준·모델·보조 조건·
  Q2 전체·경계 민감도) 탐색적 CI 보고. Holm 최악 α/4 승수 3.339 기준 Q1a 필요 문항 d=0.3
  87→≈124(164 이내), d=0.2 196→≈279(LCB 필요) — §5 7번 사이징에 반영, §4.5.1–4.5.3 비보정
  표는 탐색 분석용으로 유효 명시.
- **§2.10 + A.3 (TRACER):** §5 5단계 재작성 — 대상 코퍼스를 Olmo3로 명시(실행 가능한 유일
  형태), 공개 코드 부재→부록 명세 기반 재구현, 검색 선행 단계(n-gram/BM25 top-k), 코퍼스 검색
  ground truth의 재구현 충실도 측정 겸용. §6에 "TRACER는 재구현" 위협 신설(비공개 코퍼스
  arm으로의 정확도 전이는 상속 가정임을 명시).
- **§2.4 (미인용 2504.14655):** §2.2에 LeetCodeDataset 인용 추가(동일 시간 분할 원리의 독립
  사례로 언급) — 제거 대신 인용 채택.
- **§2.5/§2.7/A.1 (References 제목 3건):** 2511.12116 전체 제목 복원, 2410.16454 정식 제목
  기입(*Catastrophic Failure of LLM Unlearning via Quantization*, ICLR 2025, 캐비앗 해제),
  2605.24079 정식 제목 기입. ※ 검토 부록 B.4의 "같은 날 수정 완료" 기록이 사실과 달랐음을
  발견, B.4에 정정 주석 추가 (§3.4 규율의 자기 적용).
- **§2.6 (한정어):** 초록·§1의 21%→83%에 "utility-constrained unlearning methods" 한정어 추가
  (원문 자신의 한정임을 병기).
- **§2.11 (그림 미참조):** 내부 작업 산출물로 유지 결정 — 논문 무변경. **§2.12 (반올림):**
  §4.5.1의 87/196은 CLAUDE.md §4 고정값이므로 유지, §4.5.6의 신규 수치는 최근접 반올림.

영/한 동시 반영, 헤더 36=36, 신규 수치 grep 대조(0.293 3:3, 555 2:2, 3.339 1:1, 124 5:5,
279 3:3, Holm 2:2, utility 한정어 2:2) 완료.

### (f) 후속 반영 (같은 날): Instruct 변형 확정

사용자 결정: 다섯 arm 모두 instruction-tuned(-Instruct) 체크포인트로 확정. 근거: 측정 대상이
instruction 프롬프트 하의 코드 생성 pass@1, §4.5.3 예시 기저율이 instruction-tuned 수치,
LLMLagBench 검증(거부율 추적 Q&A 프로빙)의 대상도 instruct 체크포인트. §4.1 표의 모델명을
-Instruct로 갱신하고, 축약 표기가 Instruct 체크포인트를 가리킨다는 주석을 신설. 파생 수정 1건:
instruct 체크포인트는 post-training 데이터로도 벤치마크를 흡수할 수 있으므로 §5 5번의 Olmo3
ground-truth 검색 범위를 "사전학습 코퍼스"에서 "사전학습 + post-training 셋(둘 다 공개)"으로
확장. `pipeline/src/qcd/models/registry.py`의 이름·repo id 동기화 (repo id는 여전히 실사용 전
확인 필요한 placeholder). 영/한 동시 반영.

### (g) 8차 검토 오류 1·2 반영 (2026-08-05): §3.0 검정력 문장, §4.5.3 상관-표본수 범위

`review/review_findings_round8.md` §1의 확정 오류 2건. 반영 전 독립 재검증 완료
(§3.2 규율): Hanley–McNeil SE 재계산으로 n=164 → 검출 한계 0.0509, n=170 → 0.0500;
n = 785×(1−r)로 r=0.6→314, 0.8→157, 0.9→78.5.

- **§3.0:** "Q1b can detect a 0.05 AUC difference at n=164" — §4.5.2 자신이 명시한
  한계(0.051, "just short", 정확해 170)와 모순되던 문장을 "0.051 ... (detecting exactly
  0.05 requires 170 items; §4.5.2)"로 정정. §3.0과 §4.5.2가 이제 같은 값을 말한다.
- **§4.5.3:** "r이 0.6–0.9까지 갈 개연성 → 요건 157–314" — 157–314는 r∈[0.6, 0.8]에만
  해당(r=0.9는 78.5). 개연성 범위(0.6–0.9)는 실질 주장이므로 유지하고 숫자를 문서 전반의
  동일 공식(785×(1−r)) 그대로 **≈79–314**로 정정.

영/한 동시 반영. grep 검증: 구 문자열("157–314", "0.05 ... at n=164", "차이 0.05를 검출")
잔존 0건, 신규 수치 영/한 대조 0.051 3:3, 170 3:3, 79–314 1:1, 157 0:0.

### (h) §5 5번 open-data 라벨링 방법 구체화 + 제목 2건 확정 (2026-08-09)

arXiv:2404.00699 Figure 1의 open-data 분류를 §5 5번에 적용. 기존 서술은 "n-gram 및 의미적
매칭"이라는 한 구절뿐이어서 방법이 특정되지 않았다.

- **세 계열을 모두 실행하도록 명시** — (i) instance-level 문자열 매칭, (ii) 표면+AST 의미 매칭
  (arXiv:2403.04811), (iii) 패러프레이즈 탐지(arXiv:2311.04850, post-training 셋 중점).
  하나만 고르지 않는 이유를 근거와 함께 기록: **Olmo3 사전학습 혼합 자체가 벤치마크 테스트셋에
  대한 명시적 decontamination을 거쳐 구성**되었으므로 (i)만 돌리면 코퍼스 제작자의 필터가 이미
  제거한 것을 다시 재게 되어 *e* ≈ 0이 나오고, 그 값이 §4.5.2 표의 낙관적 행으로 표본 수 계획에
  전파된다. (i)은 하한선으로만 보고하고, 채택 *e* 는 완료된 것 중 가장 강한 방법의 값으로 하되
  평균내지 않는다. (i)과 (ii)–(iii)의 격차 자체를 기술적 결과로 보고 — §2.4의 78% 의미적 중복
  (arXiv:2602.12413)이 큰 격차를 예측하므로, 격차가 작으면 그쪽이 보고 가치가 있는 결과다.
- **§2.3 보강:** arXiv:2403.04811이 효과 크기 출처로만 소개되어 있어, §5가 이 논문을 방법론
  출처로 인용하는 근거가 Related Work에 없었다. 방법론적 역할을 한 문장으로 추가.
- 인프라는 미해결 항목으로 명시: suffix-array/FM-index 검색(후보 구현 infini-gram)에서 Olmo3
  코퍼스 릴리스용 공개 인덱스 존재 여부, 그리고 arXiv:2403.04811이 탐색한 규모(The Pile 380B)와
  Olmo3 코퍼스 규모의 차이에 따른 전수 AST 탐색 비용.

**제목 2건 arXiv 원문 대조로 확정** (§4.3 액션 해소):

| arXiv | 구 표기 | 확정 |
|---|---|---|
| 2311.04850 | *Rethinking Benchmark and Contamination with Rephrased Samples* | *Rethinking Benchmark and Contamination **for Language Models** with Rephrased Samples* |
| 2403.04811 | *Quantifying Contamination in Code Generation Evaluation* | *Quantifying Contamination in **Evaluating** Code Generation **Capabilities of Language Models*** (ACL 2024) |

2311.04850의 "제목 축약 가능성" 단서 문구는 제거. 2409.09927과 2505.20276은 **미확인 상태 유지**.

**미결 (사용자 판단 대기):** Dolma 3의 명시적 벤치마크 decontamination이 Olmo3 arm의 오염 의심
조건 자체를 약화시킬 수 있다는 위협을 §6에 별도 항목으로 추가할지. 이번 반영에서는 §5 5번의
방법 선택 근거로만 서술했고 §6은 건드리지 않았다.

### (i) Qwen2.5 cutoff 출시일 경계 확정 (2026-08-19)

Qwen2.5에 명확한 공식 cutoff 선언이 없어 공식 출시일을 보수적 경계로 확정했다.

- Qwen 공식 발표 페이지에서 Qwen2.5의 출시일을 **2024-09-19**로 확인했다.
- 일 단위 날짜 라벨에서 출시 당일의 모호성을 제거하기 위해 Qwen 경계의 LCB-post 적격 시작일은
  **2024-09-20**으로 고정했다.
- Qwen2.5의 주 분석 경계는 출시일 상한으로 동결한다.
- 전체 공통 LCB-post 경계는 기존 규칙대로 `max(Qwen2.5 출시일 상한, Olmo3 코퍼스 종료일)`이다.
  따라서 Olmo3 종료일이 더 늦으면 최종 주 분석 경계는 Olmo3가 결정한다.
- 현재 LCB release_v6 로더로 운영 경계 2024-09-20을 적용해 재계수한 값은 pre **690** / post
  **365** / 총 **1,055**다. 이는 데이터 가용성 점검값이며 실험 결과가 아니다.

영문 정본·한국어 미러의 §4.2 및 §5 3–4번, `pre_pilot_next_steps.md`,
`pipeline_build_plan.md`, `AGENTS.md`에 동시 반영했다. 실험 결과 수치는 추가하지 않았다.

### (j) Olmo 공식 cutoff 확정 및 공통 LCB 경계 재계수 (2026-08-20)

- Olmo3-7B-Instruct와 Olmo3.1-32B-Instruct의 공식 모델 카드가 모두
  `Date cutoff: Dec. 2024`를 명시함을 확인했다.
- 월 단위 보수성을 적용해 2024년 12월 전체를 잠재 노출 기간으로 보고, 공통
  LCB-post 시작일을 **2025-01-01**로 확정했다.
- LiveCodeBench `release_v6`를 재계수한 결과는 pre **873** / post **182** / 총 **1,055**다.
  이는 데이터 가용성 점검이지 실험 결과가 아니다.
- 두 셀 모두 사전 지정한 ≥1,000 목표를 미달했으므로 Q2를 보조·신뢰구간
  분석으로 격하했다. Q1은 영향받지 않는다.
- 공개 사전학습·post-training 코퍼스 검색은 cutoff 결정이 아니라 문항 단위 오염
  ground truth 구축 단계로 남겼다.

### (k) 실행 문서 감사 및 AWQ 조건 동결 (2026-08-22)

- H100에서 검증한 실제 구현에 맞춰 네 번째 양자화 조건을 `GPTQ-int4 or AWQ-int4`에서
  **AWQ-int4**로 동결했다. 다섯 모델 모두 llm-compressor AWQ를 사용하며 GPTQ는 실험
  로스터에 포함하지 않는다. 양자화 사다리의 수준 수와 확증 검정군은 변하지 않는다.
- Instruct 체크포인트의 가능한 노출 단계를 빠뜨리지 않도록 §5 5번의 TRACER 적용 범위를
  Olmo3 사전학습뿐 아니라 공개 post-training 코퍼스까지 포함하도록 명확히 했다.
- 영문 정본과 한국어 미러를 동시에 반영하고 모델·설계 참조 CSV의 로스터와 셀 수를 동기화했다.

### (l) 코퍼스 투명성의 분석상 지위 명확화 (2026-08-22)

- 모델 간 비교 축은 크기뿐이라는 설계 규칙을 복원했다.
- Olmo의 코퍼스 투명성은 Q1b 대리 라벨을 모델 로컬 ground truth에 대조할 수 있게 하는 운영상
  선택 속성으로 한정했다. 이를 효과 조절 축이나 모델 간 대비로 해석하지 않는다.

### (m) cutoff 라벨을 모델–문항 단위로 명시 (2026-08-24)

공통 2025-01-01 경계가 모든 모델의 청정 대조군은 만들 수 있지만, 그 이전 873문항 전체를 모든
모델의 오염 의심군으로 만들지는 못한다는 내부 정합성 문제를 수정했다. 예를 들어 Qwen2.5 출시
이후이지만 Olmo cutoff 이전인 문항은 두 모델에서 같은 시간 라벨을 가질 수 없다.

- 전역 문항 라벨을 폐기하고 `possible-exposure`, `clean-by-model-cutoff`,
  `shared-clean-control`, `boundary-ambiguous`를 **모델–문항 쌍**에 저장하도록 §4.2와 §5를 변경.
- `release_v6`의 pre 873은 arm 공통 오염 셀이 아니라 공통 경계 전 **후보 외피**로 재정의.
  주 대조군은 2025-01-01 이후 182문항으로 유지하고, 의심군은 arm별 경계 이전 부분집합으로 구성.
- 첫 post-boundary 날짜를 명시: Qwen2.5 2024-09-20, Llama 주 2024-01-01 / 민감도
  2023-04-01, Olmo 2025-01-01. Llama의 2023-04-01~2023-12-31 창은 주 런에서는
  `possible-exposure`, 민감도 런에서는 `clean-by-model-cutoff`로 사전 고정.
- 시간 라벨과 코퍼스 증거를 합치지 않음. Olmo 코퍼스 축은 `confirmed-match`,
  `no-match-found`, `not-observable`로 별도 저장하며, 검색 비검출을 `clean`으로 승격하지 않음.
- Q2 모형의 변수명을 실제 관측 지위에 맞게 `contaminated`에서 `exposure_proxy`로 변경.
- §4.5.2의 단일 *e* 감쇠식이 대칭적·비차별적 라벨 반전 모형임을 명시하고, Olmo에서는 위양성률과
  위음성률을 별도 보고하도록 추가. 우연히 근접한 541/542 표현도 Q1b ≈541 대 Q2 ≈555로 정리.
- 공개 코퍼스 검색의 비검출은 완전한 비노출 증명이 아니므로 Olmo 라벨을 "정확한 ground truth"가
  아니라 방법 의존적 운영 코퍼스 참조로 한정. 이에 따라 *e*도 정확한 측정값이 아니라 방법별
  추정치로 보고한다. CDD AUC <0.6을 라벨 잡음의 단독 신호로 읽던 문장도 삭제하고, 탐지기 바닥
  실패와 라벨 오류를 구분할 수 없다고 §6의 기존 한계와 정합시켰다.

영문 정본·한국어 미러·`reference/word_dict.md`에 동시 반영. 실험 결과는 추가하지 않았으며,
873/182/1,055는 기존 `release_v6` 가용성 점검값의 해석만 바로잡았다.

### (n) Q1b를 시간 대리 라벨 AUC로 한정 (2026-08-27)

Olmo 코퍼스 검색에서 `no-match-found`가 비노출을 증명하지 않는데도 이 결과로 시간 라벨의
오류율 *e*, 위양성률, 위음성률을 추정할 수 있다고 서술한 내부 모순을 제거했다.

- Q1b의 추정 대상을 `possible-exposure`와 `shared-clean-control`의 **대리 라벨 AUC**로 한정하고,
  실제 오염 ground truth AUC로 제시하지 않음.
- *e*=0–30% 표는 실제 오염 AUC로 재해석할 경우의 구성 타당도 민감도 분석으로만 유지하며,
  어떤 arm도 특정 행에 경험적으로 배정하지 않음.
- Olmo 코퍼스 검색은 `confirmed-match` / `no-match-found` / `not-observable`을 기술적으로 보고하되,
  이진 오류율·위양성률·위음성률을 산출하지 않음.
- 파일럿 사이징 입력을 (a)–(d) 네 값으로 축소하고, Olmo 코퍼스 결과는 별도 기술적 검증 증거로
  분리함.
- 파이프라인의 합성 오류율 계산, `olmo3_proxy_label_error_rate` 필드와 관련 테스트를 제거하고,
  변경된 `pilot_summary.json` 형식을 명시하기 위해 스키마 버전을 2→3으로 올림. 코퍼스 검색과
  TRACER 증거 저장 경로 자체는 유지함.

영문 정본·한국어 미러·`reference/word_dict.md`·`AGENTS.md`와 실행 파생 문서
(`pre_pilot_next_steps.md`, `pipeline/OLMO_GROUND_TRUTH.md`, `pipeline_build_plan.md`,
`reference/experiment_models.csv`) 및 파일럿 구현에 동시 반영. 실험 결과는 추가하지 않았다.

### (o) 시간 대리 라벨과 코퍼스 상태의 구현 정합화 (2026-08-27)

정본에서 확정한 분석 단위와 해석 제한을 실행 코드에 반영했다.

- 등록보고서가 아닌 설계를 `registered report 방식`으로 부르던 표현을 삭제하고, 실행 전
  사전 지정 관찰 연구 설계로만 기술했다.
- 모델별 첫 post 경계를 레지스트리에 고정하고, 모든 `(model, item)` 쌍에 주·민감도 시간 라벨과
  `boundary_ambiguous`를 기록하는 `model_item_labels.parquet` 생성을 추가했다.
- Q1b는 각 모델의 `possible-exposure` 대 공통 `shared-clean-control` 셀만 사용하며,
  `clean-by-model-cutoff` 중간 창을 분석 셀에서 제외하도록 집계기를 변경했다. Llama 민감도 경계에
  노출 의심 문항이 없으면 추정 불가 상태를 명시한다.
- Q2 모형과 집계 키를 `contaminated`에서 `exposure_proxy`로 변경하고, LCB 주·민감도 셀을
  모델–문항 라벨에서 구성하도록 수정했다. 재감사에서 §3.1에 남아 있던 HumanEval 대 LCB 주
  대비 표를 발견해, 모델별 LCB `possible-exposure` 대 공통 LCB control을 주 대비로 통일했다.
  HumanEval/MBPP+는 풀링하지 않는 별도 탐색적 대비로 명시했다.
- Q2 검정력 근거도 현재 주 대비에 맞춰 재계산했다. 보수적 비페어링 p=0.5에서 공통 대조
  n=182와 의심 n→∞의 MDE는 14.7%p, 의심 n=873일 때는 16.1%p다. HumanEval n=164의
  15.5%p는 보조 대비의 최선값으로만 유지했다.
- Q1b 집계가 모델–문항 라벨을 도입하면서 HumanEval/MBPP+의 `possible-exposure` 행까지 포함할
  수 있던 구현 오류를 재검토 중 발견했다. Q1b 입력을 LCB 행으로 먼저 제한하고, HumanEval
  행을 포함한 회귀 테스트로 재발을 막았다.
- 코퍼스 검색 출력을 `confirmed-match` / `no-match-found` / `not-observable`의 3상태와
  `coverage_complete`로 구분했다. 부분 스캔과 개별 shard 비검출은 `not-observable`이며, 전체
  shard가 완료된 최종화 단계에서만 `no-match-found`를 생성한다. 약한 부분 일치 후보도 전체
  스캔 최종화 뒤 임계값을 넘지 못하면 `no-match-found`가 되도록 고쳤고, 증거 스키마를 3으로
  올려 기존 shard 결과와의 혼합을 차단했다.
- CDD 관문은 시간 대리 대비에서의 운영적 탐지기 선택 규칙으로 한정했다. 관문 실패는 검증된
  실제 오염에 대한 CDD의 작동 불능 증거로 보고하지 않는다.
- 재감사에서 고정 0.7936 관문이 양성·음성 각 542문항을 전제해 HumanEval/MBPP+ 풀링 금지 및
  실제 LCB 셀 크기와 충돌함을 확인했다. 고정 임계값을 폐기하고 각 7B 파일럿 arm의 실측 AUC가
  함의하는 가정 ΔAUC를 실제 n+/n−·실측 r의 대응 AUC 검출 한계와 비교하도록 변경했다.
  `release_v6` 모델–문항 라벨을 재계수해 Qwen 690/183/182, Llama 326/547/182,
  Olmo 873/0/182(`possible`/`clean-by-model-cutoff`/`shared`)를 확인했다.
- 관문 실패의 분석 파급 범위를 C4 Q1b로 제한했다. 노출 라벨을 사용하지 않는 C1 Q1a는
  유지하고, 두 7B arm 중 하나라도 실패하면 C4는 대체 검정 추가 없이 추정 불가로 보고한다.

영문 정본·한국어 미러·용어집·실행 계획·파이프라인 문서·테스트를 함께 갱신했다. 변경은 분석
계획과 저장 스키마의 정합화이며, 새 실험 결과를 포함하지 않는다.

### (p) 과학적 파일럿 폐기와 구현 검증 경계 고정 (2026-08-28)

H100 대여 전의 소규모 실행은 저성능 머신·제한된 실제 하드웨어에서 구현을 검증하기 위한
절차였으나, 이전 문서는 그 출력에서 효과 크기·AUC·기저율·상관을 계산하고 검정력과 CDD의
확증 자격을 바꾸는 과학적 파일럿으로 확대해 서술했다. 이 혼선을 제거했다.

- 로컬 합성 드라이런과 제한된 H100 스모크 테스트를 구현 검증 전용으로 한정했다. 확인 대상은
  모델 로딩, 양자화 호환성, 스키마, 유한 로그확률, 샌드박스, 메모리와 처리량이다.
- 구현 검증 출력은 논문 근거, 효과 크기, AUC, pass rate, 상관, 검정력 재산정, 탐지기 순위 또는
  C1–C4 자격에 사용하지 않는다. 결과 확인 전 실행 실패가 드러난 경우에만 운영 설정을 바꾸고,
  변경 기록 후 본 실행 전에 다시 고정한다.
- 데이터 의존적 CDD 관문을 폐기했다. C1–C4는 본 실행 전에 고정하며, CDD가 본 연구에서 우연
  수준이어도 사후 제거하지 않고 추정치와 신뢰구간을 연구 결과로 보고한다.
- `run_main.py`만 연구 데이터 생성기로 남겼다. `run_pilot.py`와 `aggregate_pilot.py`는 실행을
  거부하는 폐기 호환 진입점으로 바꾸고, CDD 관문 구현을 삭제했다. 합성 집계 보조 코드는
  개발 전용 상태와 파일명으로 격리해 논문 증거가 아님을 스키마에 명시했다.
- §4.6–§4.7, §5 실행 순서, 타당성 위협과 한계를 영문 정본·한국어 미러에서 함께 수정하고,
  `AGENTS.md`, 실행 로드맵, 파이프라인 README·구축 계획, 모델 표와 용어집을 동기화했다.

이 변경은 (n)–(o)의 파일럿 사이징·CDD 관문 부분을 폐기·대체한다. 기존 H100 스모크 출력은
구현 검증 부산물로만 남으며 논문 결과에 포함하지 않는다. 새 실험 결과는 추가하지 않았다.

### (q) 원고 오류 재검증과 분석 정의 명시 (2026-09-09)

사용자의 “더블 체크 하고, 그래도 여전히 오류라고 생각되면 수정” 요청에 따라 앞선 검토를
원문·현재 구현·독립 수치 계산으로 재검증했다. 영어 정본, 한국어 미러와 현재 지침·용어집을
동기화했다. 과거 `review/` 기록은 수정하지 않았다. 다음은 연구 결과가 아닌 검토 판정과
실행 전 설계 명시이다.

| 재검토 항목 | 판정과 반영 |
|---|---|
| CDD의 짧은 출력 거리 임계값 | **오류 유지·수정.** Sela §3.1 식 (1)의 l은 greedy와 모든 샘플의 실제 최대 길이이며 100은 절단 상한이다. 코드의 `alpha*100`을 `alpha*actual_max_length`로 수정했다. 10토큰·1 edit 반례와 19/20토큰의 전체 최대 길이 회귀 테스트를 추가했다. |
| Q2의 DiD와 회귀 교차효과 부호 | **오류 유지·수정.** Q=1 양자화, E=1 노출 가능이면 β_QE=−J이다. 확률 셀·조건부 logit·주변 집계량을 구분했다. 0.85→0.83 및 0.35→0.31에서 J≈−0.032106, 회귀 방향 대비는 +0.032106이다. |
| 기저율 아티팩트 “항상 음수” | **일반화 오류 유지.** 기존 0.85/0.35 표는 재현되므로 유지했다. 같은 정규 난이도 SD=1.5, β=0.5에서 기저율 0.60/0.10의 하락은 8.7218/3.1527pp로 교차효과 +5.5691pp이다. 약 0.023의 주변 logit 편향도 해당 예제의 값이지 보편 상한이 아니다. |
| 표본 수의 1문항 차이 | **이전 지적 완화.** 반올림한 계획 근사 자체를 계산 오류로 보지 않는다. 87/196, 170/287/541/1,268 표는 유지하고 근사임을 명시했다. 엄밀한 정수 올림은 각각 88/197, 171/288/542/1,268이다. 그림의 반올림 수치를 바꾸지 않았다. |
| AUC 표와 라벨 잡음 공식 | **가정 누락 유지·수정.** AUC=0.70, 집단당 n, 두 AUC 추정량의 상관 r, Hanley–McNeil SE 경로를 명시했다. (1−2e)는 실제 양성 비율 0.5와 점수에 독립적인 대칭 반전이라는 가상 모형에 한정했다. 실제 양성 비율 0.2, e=0.2이면 일반 감쇠 계수는 0.441176이다. |
| 785의 “가정 없는 상한” | **오류 유지·수정.** 독립 Bernoulli 셀·p=0.5·같은 셀 크기·원시 %p·정규근사의 채택 계획 기준으로 유지했다. 대응 로그오즈 GLMM의 직접 검정력이 아니다. |
| CDD AUC≈0.5이면 Q1b 불가능 | **논리 오류 유지·수정.** 다른 탐지기의 AUC는 이동할 수 있다. 예를 들어 CDD 0.5→0.5, 확률 탐지기 0.7→0.4이면 상대 순위가 역전된다. 우연 수준은 AUC의 수학적 최솟값도 아니다. |
| C1–C4 미지정 요소 | **실행 전 명시.** 사용자가 판단을 위임했고 표의 기존 Primary 역할을 따라 Qwen2.5-32B-Instruct로 한정했다. C1–C3은 전체 LCB 1,055문항, C4는 690/182 대리 집단이다. 확률 계열은 두 AUC의 동일 가중 평균이다. 간격 변화와 실제 순위 역전을 구분하고 방향별 교집합–합집합 검정, 양방향 보정 및 Holm을 명시했다. 두 AUC 계획 표가 C4 검정력을 입증하지는 않는다. |
| Llama 민감도 분석 | **오류 유지·수정.** 2023-04-01 경계의 LCB 노출 가능 문항은 0이므로 자체 대비와 경계 이전 분포 비교는 추정 불가다. 풀링 분석의 모델 모집단 변경을 명시하고 계산 불가한 결과에서 강건성을 주장하지 않는다. 지식 경계는 실제 학습 cutoff의 검증이 아니며 체계적 시간 라벨 오분류가 무해한 감쇠만 보장하지 않는다. |
| 혼합효과·불확실성 표현 | **오류 유지·수정.** 랜덤 절편만으로 모델별 기울기와 모든 의존성을 흡수하지 못한다. 현재 `fit_vb`의 출력은 변분 사후평균·SD이며 근사 신용구간으로 보고해야 한다. 비유의성은 가정 검증이나 동등성 입증이 아니다. |
| 선행연구와 메커니즘 | **지적 유지·수정.** 양자화×membership inference 및 verbatim extraction 선행연구 두 편을 §2.8에 추가했다. 유틸리티 제약 unlearning의 억제 역전이 자연 암기 답의 선택적 소실을 입증하지 않는다고 한정했다. Sela 결과는 70M–410M 실험 범위에 한정했다. |
| 기타 과장·명세 누락 | **수정.** 77.5%의 분모, AST 유사도와 의미 등가성, 아키텍처 통제 주장, nf4 최대 효과 단정, double quantization 무효과 단정, 잘못된 절 참조와 참고문헌 제목을 정리했다. CDD n=50/alpha=0.05/상한100, Min-k=20%, 고정 프롬프트 점수와 AWQ 보정 명세를 추가했다. |

**수치 재계산 경로.** 표준정규 분위수의 합은 2.801585이다. Q1a는 n=(z_0.975+z_0.8)²/d_z²이며
근은 87.2098/196.2220이다. AUC는 A=0.70, Q₁=A/(2−A), Q₂=2A²/(1+A),
SE²=[A(1−A)+(n−1)(Q₁−A²)+(n−1)(Q₂−A²)]/n²를 사용했다. n=164의
SE=0.02873235, r=0.8의 MDE=0.05091022이다. 잡음 행은 A를 0.5+(1−2e)(A−0.5)로,
목표 ΔAUC를 0.050(1−2e)로 바꾸어 같은 공식·이분 탐색으로 풀었다. 연속 근은
170.0051/287.3257/541.3706/1267.8944이다. 기저율 반례는 표준정규 난이도를 수치 적분하고
각 목표 주변 확률에 맞는 절편을 이분 탐색한 뒤 동일 β를 빼서 계산했다. 모두 계획 가정에
대한 계산이며 모델 실행 결과나 실측 검정력 재산정이 아니다.

**원문 및 구현 대조 근거.**

- [Sela, arXiv:2603.03203, §3.1 식 (1)](https://arxiv.org/html/2603.03203v3): 로컬 PDF와 본문에서 CDD 실제 최대 길이 확인.
- [Catastrophic Failure of LLM Unlearning via Quantization](https://arxiv.org/abs/2410.16454): 로컬 PDF의 유틸리티 제약·21%→83% 문맥 유지.
- [LLMLagBench](https://arxiv.org/abs/2511.12116): 로컬 PDF의 행동적 지식 경계와 선언 cutoff 구분.
- [How Quantization Impacts Privacy Risk on LLMs for Code?](https://arxiv.org/abs/2508.00128), [Bits and Memories](https://arxiv.org/abs/2607.25451): 원문 PDF 및 arXiv 제목·대상·측정량 대조.
- [Soft Contamination Means Benchmarks Test Shallow Generalization](https://arxiv.org/abs/2602.12413): 77.5%의 벤치마크 문제 분모와 top-100 검색 조건.
- [statsmodels fit_vb 공식 문서](https://www.statsmodels.org/stable/generated/statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM.fit_vb.html)와 `pipeline/src/qcd/analysis/mixed_effects.py`: 평균장 변분 추론과 구간 의미 확인.
- [DeLong et al., correlated ROC AUC comparison](https://pubmed.ncbi.nlm.nih.gov/3203132/): 같은 문항의 다중 AUC 공분산을 사용하는 근거. C4의 두 방향 조합은 본 프로토콜의 사전 지정이며 이 논문의 기존 실험 결과가 아니다.

**검증·구현 경계.** CDD 정의를 `actual-max-truncated-length-v2`로 실행 명세의 해시에 포함해
이전 점수 폴더에 새 점수를 이어 쓰지 못하게 했다. 기존 스모크 출력은 과거 구현 검증 자료로만
남긴다. 전체 로컬 테스트 명령
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=pipeline/src pipeline/.venv/bin/python -m pytest pipeline/tests -q -p no:cacheprovider`
결과는 **232 passed, 2 skipped**이다. torch 의존 모듈 두 개는 로컬 환경에서 건너뛰었다.
새 GPU 실험이나 연구 데이터 생성은 수행하지 않았다.

§4.5.6의 C4 전체 검정·보고 조립은 기존 pairwise AUC helper만으로 완료되어 있지 않다.
Q2 모델별 적합·명시적 기준 코딩·진단 및 구간 보고 연결, AWQ 보정 중복 검토·최종 산출물 고정도
실행 전 과제로 명시했다. 이번 수정은 이 기능들의 구현 완료를 주장하지 않는다. 코드의 확인된
CDD 오류와 재개 호환성은 수정했으며, 새로운 분석 파이프라인 전체를 추가한 변경은 아니다.

### (r) (q) 변경점 자체의 재검토 (2026-09-09)

사용자의 변경점 더블 체크 요청으로 (q)의 diff, 수치 예제, CDD 원문 식, 현재 채점 구현 및
영·한 대응을 다시 검토했다. 직전 변경에서 새로 생겼거나 충분히 제거되지 않은 문제를 정정했다.

1. **로그오즈 집계 편향의 β 오기.** (q)에서 본문·미러·용어집에 0.023의 예제를 β=0.5로
   명시한 것은 잘못이었다. 기저율 0.85/0.35, SD=1.5에서 β=0.5는 0.0204697849,
   β=0.75는 0.0230500915이다. 두 값을 모두 명시했다. 기존 격자 적분 구현과 독립적인
   100점 Gauss–Hermite 적분·절편 이분 탐색으로 대조했다. (q)의 0.60/0.10 반례는 별개이며,
   그 원시 %p 교차효과 +5.5690705는 재현된다. 기존 `test_logodds.py`도 0.023의 β가
   0.75임을 검사하고 있었다. 테스트 통과만으로 문서의 조건 표기가 검증되지는 않았다.
2. **J 해석의 잔존 척도 혼용.** J를 조건부 로그오즈로 재정의하고도 §3.1의 하락량 설명과
   −2pp 예제를 남겨 같은 척도처럼 읽힐 수 있었다. A=0.85, C=0.35와 로그오즈 하락
   0.30/0.20이면 B=0.8076172, D=0.3059676, J=+0.10이지만 원시 교차효과는
   −0.1649554pp이다. 부호별 설명을 조건부 로그오즈로 한정하고 이 반례를 넣었다. near-zero
   추정치가 효과 차이 부재를 입증한다는 잔존 문장도 제거했다. 노출 확인과 B-vs-A 증가만으로
   unlearning 역전을 입증하지 못하고 사전 억제와 그 역전 근거가 필요하다고 명시했다.
3. **CDD 점수의 이산성.** 한국어판의 “0–1 연속 peakedness”는 잘못된 분류였다. n=50에서
   가능한 점수는 {0, 1/50, …, 1}이며 영·한 양쪽에 이를 명시했다. CDD의 실제 최대 길이
   계산 자체는 [원문 §3.1 식 (1)](https://arxiv.org/html/2603.03203v3)과 일치한다.
4. **C4 퇴화 분산 문구 명확화.** 제외 사유를 두 격차 중 어느 하나의 0/미정의 분산으로
   한정했다. 전체 여섯 AUC 공분산 행렬이 특이하더라도 선형 대비의 분산이 양수이면 계산할 수
   있으며 역행렬은 필요하지 않다. CDD 점수 상수만으로 C4를 제외하지 않는다. 방향별 max
   p값(교집합–합집합), 두 방향 선택의 계수 2, 최종 Holm의 구조는 유지한다. 이는 점근적
   검정 계획의 검토이며 C4 실행기 구현·경험적 검정력 검증 완료를 뜻하지 않는다.

이번에는 분석·생성 코드를 변경하지 않았다. 관련 기존 테스트 네 파일(detectors, real_run,
logodds, power)을 실행해 **64 passed**를 확인했다. 최초 호출은 저장소에 없는 AUC 테스트
파일명 때문에 수집 전 실패했으며, 실제 파일 목록을 확인한 후 위 네 파일을 실행했다.
AUC 계획 표는 별도 재계산으로 대조했다. 영·한의 새 수치·부호식·문항 수·arXiv ID·절 구조가
일치하며 `git diff --check`도 통과했다. C4 실행기와 Q2 보고 연결 등 (q)에 명시한 구현 과제는
여전히 별도 실행 전 과제다. 연구 데이터 생성이나 새 GPU 실행은 없었다.

### (s) 주장·검정 대응과 잔존 용어의 재점검 (2026-09-10)

사용자의 재검토 요청에 따라 (q)–(r)의 핵심 수치와 부호를 독립 재계산했다. 잡음 표의
정수 최소값 171/288/542/1,268은 각각 n에서 목표를 충족하고 n−1에서는 미달한다.
100점 Gauss–Hermite 적분으로 β=0.5의 0.0204697849, β=0.75의 0.0230500915를 다시
확인했다. C4의 방향 조합은 같은 부호·반대 부호·0인 경계에서 의도대로 동작한다는 대수적
점검을 수행했다. 이는 이상적 정규근사의 식 점검이며 미구현 DeLong 실행기의 검정력·보정
성능을 검증한 것은 아니다. 이 범위에서 추가적인 계산·CDD 코드 오류는 발견하지 않았다.

다음은 남아 있던 설명의 정합성을 보완한 사항이다.

- **Q1a에서 주장 가능한 범위.** C1–C3은 탐지기마다 평균 점수 이동을 0과 비교하며, 탐지기 간
  효과 차이를 직접 검정하지 않는다. 유의/비유의 차이만으로 서로 다른 변조를 주장해서는 안
  된다. 원시 점수 척도도 다르다. C4는 동일 가중 평균 확률 AUC의 순위 역전이며 두 확률
  탐지기 각각의 역전은 아니다. 이 한계를 §4.5.6과 용어집에 명시했다. 추가 대비는 탐색적으로
  유지하고 C1–C4·모델·문항 집합은 바꾸지 않았다. 근거는
  [Gelman & Stern (2006)](https://sites.stat.columbia.edu/gelman/research/published/signif4.pdf).
- **시간 분할의 인과적 지위.** §2.2와 용어집이 cutoff 분할을 본 연구의 자연 실험으로
  일반화하던 문구를 한정했다. [Roberts et al.](https://arxiv.org/abs/2310.10628)의 원문은
  실제로 자신들의 GPT cutoff 활용을 natural experiment라고 부른다(초록과 로컬 PDF 대조).
  그 귀속은 보존하되, 우리 모델별 LCB 시간 대리 비교의 무작위성·인과 식별을 자동으로
  정당화하지 않는다고 명시했다. 서론의 Q2 소개에도 실제 오염/청정 대신 대리 조건과 조건부
  로그오즈 대비를 명시했다.
- **용어집의 잔존 오류.** 신뢰구간의 반복 표집 포함률과 Q2의 근사 사후 신용구간을 구분했다.
  비이진 점수가 모두 연속형이라는 설명, 부분 테스트 점수의 검정력 향상 보장, 실제 오염을
  이미 관측했다는 설명을 수정했다. 등가성 마진이 현재 지정되지 않았음을 맞췄다.
  `AGENTS.md`의 “모든 arm의 의심 n이 873보다 작다”는 잔존 문구도 ≤873으로 맞췄다.

이번 추가 변경은 문서에 한정했다. 관련 기존 테스트 네 파일은 **64 passed**였고,
영·한 핵심 수치·부호식·문항 수·arXiv ID 및 `git diff --check`를 확인했다.
C4 분석 실행기와 Q2 보고 연결 등 앞서 명시한 실행 전 구현 과제는 완료 상태로 바꾸지 않았다.
