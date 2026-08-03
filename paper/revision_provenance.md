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
