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
    identifies as a precondition for Q1b. This updates the design rule previously stated as "size is the
    only remaining axis across models": architecture is still not an axis (all arms are dense
    transformers), but **training-corpus transparency** now joins size as a deliberate second axis, adopted
    to serve §4.5.2. Consequential edits: §2.4 (arXiv:2602.12413's Olmo3-corpus finding becomes a direct
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
  `fig_round4_corrections.png` remain valid.

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
§5 4단계에 LLMLagBench 실행 경로(리더보드/평가 요청/Olmo3 코퍼스 메타데이터) 및 거부율 실패
모드(Qwen2.5-Omni 87% 거부) 명시; §5 6단계에서 70B 대여 창 결정 제거; §6에 "규모 불일치
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
arm에도 방어 가능한 경계가 존재하며, LLMLagBench 요청이 무응답/비결정이어도 설계가 자립한다.
파생 수정: §5 3단계(최악 경계 기준 집계), §5 4단계(폴백 문구), §6(증거 이질성 위협 확장),
§8(arm 의존적 정밀도 한계 신설). 영/한 동시 반영, 핵심 어구 grep 대조 완료. CLAUDE.md §7
5단계 동기화.

### (d) 후속 반영 (같은 날): 경계 민감도 분석 사전 지정

사용자 제안("Llama를 선언 cutoff와 추정 cutoff로 두 번 테스트")을 채택하되 일반 규칙으로
확장해 반영. §4.2에 신설: 코퍼스 등급 미만의 모든 arm은 라벨 의존 분석(Q1b/Q2)을 상·하한
경계 양쪽으로 재실행 — Llama-3.1-8B는 선언 2023-12(주) vs 탐지 2023-03(민감도), Qwen2.5는
출시일 상한 vs 향후 관리자 추정치. 역할(주/민감도)은 사전 고정, 결과를 본 뒤 재조정 금지.
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
