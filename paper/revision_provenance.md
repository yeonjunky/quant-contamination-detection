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

No further review rounds have been run against this consolidated document; a seventh adversarial pass before
execution begins is recommended, particularly to check that this synthesis introduced no new
transcription errors when merging the two source documents' independently-corrected numbers.
