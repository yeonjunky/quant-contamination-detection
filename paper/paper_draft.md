# Does Quantization Erase the Evidence? Contamination-Detection Signals Under Post-Training Quantization in Code LLMs

*[Working title — subject to revision after results are in]*

**Status:** Research design draft (pre-execution). Sections marked **[TBD — pending execution]** are
placeholders to be filled in after the plan in §5 is carried out; nothing below reports actual
experimental results.

Authors: [TBD] · Affiliation: [TBD]

---

## Abstract

Reported accuracy drops from post-training quantization are usually interpreted as a loss of model
*capability*. We argue that part of this drop may instead be a loss of *memorized answers*:
if a benchmark problem was seen during pretraining, a model's correct response may reflect recall
rather than reasoning, and quantization is known to perturb memorized traces disproportionately
(unlearned knowledge has been shown to resurface after 4-bit quantization, rising from 21% to 83%
retained in one study; the mechanism has been formalized as a sparsity-permanence tradeoff in which
sub-threshold parameter changes are erased by quantization's bin structure). Separately, contamination-detection
research shows that detectors are not interchangeable: output-distribution-peakedness detectors (CDD)
require verbatim memorization and collapse to chance without it, while probability-based detectors
(perplexity, Min-k% Prob) remain informative under the same conditions. No published work has examined
what happens at the intersection: does post-training quantization differentially reshape these two
families of contamination signal, and does it change the outcome of contamination-vs-clean comparisons
on code-generation benchmarks?

We present a registered-report-style observational design (analysis plan fixed before execution, though
not lodged with an external registry) to answer this at a scale (7B–70B) roughly one to three
orders of magnitude larger (17×–1,000×, depending on which endpoints are compared) than the only prior
contamination-detection study to probe this failure mode (70M–410M). The design's primary question (Q1) compares peakedness- and probability-based detection
signals across quantization precision on a paired, per-item basis, which is well powered even at
benchmark-imposed sample-size ceilings (e.g., 164 items). A secondary question (Q2) asks whether the
quantization-induced pass@1 drop itself differs between contamination-suspect and clean-benchmark
conditions, analyzed on the log-odds scale via mixed-effects logistic regression to avoid a base-rate
confound we identify and quantify (§4.5.3). We report power calculations, a pilot go/no-go gate for the
weaker detector (CDD), and an explicit statement of the design's limits as an observational rather than
causal study. **Results: [TBD — pending execution of §5].**

---

## 1. Introduction

Post-training quantization (PTQ) is now standard practice for deploying large language models, and a
substantial literature reports its effect on downstream accuracy. That literature treats accuracy drops
as measurements of degraded *capability*. This framing has an unexamined assumption: that benchmark
performance under full precision reflects capability rather than recall of memorized training data.

Two independent lines of evidence suggest this assumption should not be taken for granted.

**Quantization perturbs memorized traces.** Studies of machine unlearning — the practice of suppressing
specific knowledge in a trained model without full retraining — find that "erased" knowledge often
resurfaces after quantization: one study reports retained-knowledge rates rising from 21% at full
precision to 83% after 4-bit quantization (arXiv:2410.16454). A mechanistic follow-up explains why:
the parameter changes that constitute a successful unlearn are frequently 47–828× smaller than a single
NF4 quantization bin, so quantization's rounding simply erases them (arXiv:2605.15138, "Forgetting That
Sticks"). If quantization can un-erase suppressed knowledge, it is plausible that it can also erase
knowledge that was never meant to be there — i.e., memorized benchmark answers.

**Contamination detectors are not interchangeable.** A 2026 replication study of the peakedness-based
CDD detector found it collapses to chance-level accuracy whenever the underlying memorization was not
verbatim, even on data that was "detectable by simpler methods" — specifically, probability-based
detectors (perplexity, Min-k% Prob) outperformed CDD in every condition where any method exceeded chance
at all (arXiv:2603.03203, *No Memorization, No Detection: Output Distribution-Based Contamination
Detection in Small Language Models*). The paper's own limitations section cautions that this behavior
"should not be extrapolated to larger scales without further investigation" — a caution that cuts both
ways, since the paper's positive result (CDD works) was itself obtained only at 7B and is not established
at the 32B–70B scale either. What the paper does establish is a *mechanism*: CDD is governed by a sharp
memorization threshold tied to the absolute number of trainable parameters, not model size per se.

Put together, these two threads motivate a question that neither literature has asked: **if quantization
measurably disturbs memorized traces, and contamination detectors differ in how much they depend on those
traces, does quantization differentially reshape what different contamination detectors see?**

This is the paper's primary contribution. A related but harder-to-power question — whether the
*quantization-induced accuracy drop itself* differs between contaminated and clean benchmark conditions —
is a natural motivating question but, as we show in §3.3 and §4.5, is statistically under-powered at
benchmark-imposed sample sizes and is therefore treated as **secondary**. We are explicit about this
distinction throughout: the paper's contribution claim rests on the primary question (Q1), and the
secondary question (Q2) is reported with appropriately wide uncertainty rather than forced into
statistical significance.

**Contributions (stated as design commitments, to be confirmed or refuted by execution):**

1. The first measurement, at 7B–70B scale, of how post-training quantization affects the relative
   behavior of peakedness-based vs. probability-based contamination-detection signals (Q1a/Q1b).
2. A bounded, log-odds-scale estimate of the quantization × contamination interaction on pass@1 in code
   generation, reported as an association rather than a causal effect, with an explicit base-rate-confound
   correction (Q2).
3. A reusable item-level dataset — pass@1, partial credit, token log-probability, and three detector
   scores, crossed with quantization technique, precision, and model — intended to support future
   contamination-detection benchmarking work independent of this paper's own conclusions.
4. A negative/boundary result, regardless of Q1's outcome: either evidence that CDD is inoperative at
   32B–70B scale (extending arXiv:2603.03203's threshold finding beyond its tested range) or evidence that
   it is operative, both of which are currently unknown.

---

## 2. Related Work

### 2.1 Contamination: definitions, causes, mitigation (surveys)
Data contamination — benchmark items appearing, verbatim or paraphrased, in a model's training data — is
a mature research area with several existing surveys. arXiv:2404.00699 (*A Comprehensive Survey of
Contamination Detection Methods in Large Language Models*, TMLR 2025) has the broadest detection-method
coverage; arXiv:2502.14425 is the more recent general survey of definitions, causes, and mitigation but
covers detection methods less exhaustively; arXiv:2605.26133 unifies contamination with membership
inference and training-data-extraction; arXiv:2502.17521 motivates the shift from static to dynamic
benchmarking, which we invoke to justify our use of LiveCodeBench.

### 2.2 Temporal-split contamination measurement
Splitting a benchmark by public release date relative to a model's training cutoff is a long-established
natural-experiment design for contamination (arXiv:2310.10628, *Data Contamination Through the Lens of
Time*), directly justifying our LiveCodeBench pre-/post-cutoff split (§4.2). LiveCodeBench itself
(arXiv:2403.07974) implements continuous, dated problem collection specifically to support this design.
Because publicly declared training-cutoff dates can be wrong or absent, we additionally rely on
arXiv:2511.12116 (*LLMLagBench*), which estimates a model's *actual* temporal training boundary from its
knowledge of recent events, as a validation step (§5, step 4) rather than trusting declared cutoffs
outright.

### 2.3 Effect sizes of contamination
arXiv:2501.18771 provides a controlled, causally-identified estimate of contamination's effect by directly
pretraining 1B/8B models on machine-translation data with contamination injected at controlled stages,
scales, and formats. We treat this as the methodological reference point for causal identification, but
**cannot replicate its design**: it requires pretraining from scratch, and our study uses off-the-shelf
7B–70B models in the code domain. We can therefore only approximate its causal design observationally
(§6). arXiv:2403.04811 and arXiv:2506.02791 quantify contamination's effect size specifically in code
generation, with the latter noting that most prior work measures only sample-level contamination and
under-counts the more common partial-contamination case. arXiv:2507.19219 offers a one-time-pad-based
framework for quantifying benchmark-score overestimation generally.

### 2.4 Limits of contamination detection
This subsection is the paper's most load-bearing prior work. arXiv:2311.04850 shows n-gram-based
decontamination filtering is trivially evaded by paraphrase or translation. arXiv:2602.12413 (*Soft
Contamination Means Benchmarks Test Shallow Generalization*) extends this: semantic (non-lexical)
duplication is undetectable by n-gram matching and was found pervasively in the Olmo3 pretraining corpus,
including in CodeForces-derived data (78% semantic duplication reported) — this is the strongest available
objection to treating any time-filtered benchmark as "clean," and we address it directly in §6.
arXiv:2402.02823 and arXiv:2409.09927 further document that intentional contamination is easy to hide
from detectors, and that existing detectors disagree with each other on modern LLMs.

**arXiv:2603.03203** (*No Memorization, No Detection*) is the anchor citation for our primary research
question, and because its claims are easy to over-generalize, it merits a precise account of what it does
and does not establish.

- **What it shows:** Sela (Tel Aviv University) replicates the CDD contamination detector of Dong et al.
  (2024) on 70M–410M models with
  controlled contamination injected via LoRA fine-tuning on GSM8K/HumanEval/MATH, CDD collapses to
  chance-level accuracy under most conditions — even when the underlying data is "detectable by simpler
  methods." Probability-based detectors (perplexity, Min-k% Prob) outperform CDD in every condition where
  *any* method exceeds chance. The paper's strongest supporting quote for using probability-based methods
  as the primary detector family is: *"The gap is largest precisely where it matters most: at low
  contamination levels and under parameter-efficient fine-tuning, where CDD is uniformly at chance but
  probability-based methods already show signal."* We deliberately do not paraphrase this as
  "probability-based methods work wherever CDD fails." The paper's abstract phrases the comparison
  tautologically — "outperform CDD in all conditions where any method exceeds chance" does not imply
  probability-based methods always exceed chance — and while its Conclusion goes further
  (*"…including those where CDD fails entirely"*), that clause establishes only that the set of such
  conditions contains cases of outright CDD failure, not that it exhausts them. The quoted sentence above
  supports the design decision we actually need — probability-based detectors as primary, CDD as a
  comparison arm — without the stronger claim.
- **What governs CDD's failure:** not verbatim memorization *per se*, but "a memorization threshold
  [that] governs detectability," where "CDD accuracy transitions sharply from chance to >90% as
  fine-tuning capacity crosses a threshold" that "depends on the interaction of model size, adapter rank,
  and training duration." The paper attributes this specifically to *"the relevant factor is not the LoRA
  rank itself but the absolute number of trainable parameters."* The absolute-capacity comparison the
  paper draws is between the original CDD paper's positive 7B result and the replication's own small-model
  runs: *"LoRA r=8 on a 7B model yields roughly 4M trainable parameters; the same rank on our 70M model
  yields only 98K. Our LoRA r=256, which provides 3–25M trainable parameters, is closer in absolute
  capacity to what low-rank LoRA provides on 7B models, and this is where CDD begins to work in our
  experiments."* The two figures are **not interchangeable**: ~4M is the paper's own estimate for LoRA r=8
  on a 7B model, whereas 3–25M is the trainable-parameter range of the replication's *own* r=256
  configurations on 70M–410M models (Table 1: 3.1M / 9.4M / 25.2M). Note that the 7B figure appears only in
  the paper's Discussion prose; Table 1 tabulates 70M/160M/410M and does not contain it.
- **What it does not show:** anything about 32B–70B models, or about pretraining-time (as opposed to
  fine-tuning-injected) contamination. The paper explicitly states its small-model findings "should not be
  extrapolated to larger scales without further investigation." It is tempting to argue in the opposite
  direction — that CDD's positive 7B result means CDD "works" at the scales this design targets — but that
  argument violates the same caveat it would have to cite. **The position of 32B–70B models relative to CDD's memorization
  threshold is simply unknown**, and could fail in either direction: if these larger models sit above the
  threshold, CDD works and our design is well-powered (§4.5.2); if they sit at or near the threshold's
  chance-level side, CDD's AUC is ≈0.5 and **no amount of additional data will make Q1b detectable**,
  because the effect size itself — not the standard error — collapses to zero. This is why §4.6 makes
  measuring CDD's baseline AUC an explicit pilot gate rather than an assumption.

### 2.5 Code-benchmark-specific contamination
arXiv:2605.24079 (*TRACER*) models code contamination as a three-tier semantic-duplication problem
(functionally identical / nearly identical / shared-logic) and is the tool we use to measure residual
contamination in our conditions directly, rather than relying solely on the pre/post-cutoff date proxy
(§5, step 5). arXiv:2411.10842 (*CodeCleaner*) and arXiv:2503.06643 offer refactoring- and
transformation-based mitigation approaches; arXiv:2503.13572 is a domain-specific (Verilog) contamination
case study illustrating the same issues outside Python/general-purpose code.

### 2.6 Post-hoc decontamination — an alternative research direction
A separate line of work corrects for contamination without changing the benchmark itself:
arXiv:2509.15218 (LNE-Blocking) recovers pre-contamination performance estimates; arXiv:2601.19334
performs inference-time decontamination; arXiv:2605.21543 develops decontamination theory for jointly
benchmarking multiple models (directly relevant to our three-model comparison, §4.1); arXiv:2506.04142
identifies "shortcut neurons" inside contaminated models that mechanistically explain overestimation — a
promising direction for future work connecting this paper's findings to model internals, but outside the
present design's scope.

### 2.7 Quantization effects in code generation
The original citation motivating this design's power analysis (arXiv:2505.20276) reports 8-bit
quantization preserves accuracy (≈0.8% drop) while 4-bit quantization can degrade it by up to 59% —
including a 32% drop for Llama-3.1-70B under calibration-free BNB-nf4 on the same task family. This result
is for **long-context (>64K token) evaluation**, not code generation, and cannot be transferred directly;
we use it only as the effect-size reference for our BNB-nf4 arm (§4.3), where it is our best available
prior for "worst case." For code generation specifically, the consensus in three papers we newly add to
this design (arXiv:2503.07103, arXiv:2507.09665, arXiv:2506.22776) is that **calibration-based** 4-bit
quantization (AWQ/GPTQ) shows little to no significant degradation, and in one study (arXiv:2506.22776)
quantized models are *more* robust under adversarial conditions (51.59% vs. 42.86%). This consensus
narrows our expected effect size for the AWQ/GPTQ arm and — because it is itself derived largely from
benchmarks whose contamination status is unexamined — is part of this paper's motivation rather than a
reason to expect a large effect (§4.5.3, §7).

### 2.8 The gap this paper addresses
None of the above literatures intersect. The contamination-detection literature (§2.4) has not examined
quantized models. The quantization literature (§2.7) has not stratified its accuracy measurements by
contamination status. The unlearning literature (introduction) establishes that quantization perturbs
memorized traces but has not connected this to benchmark contamination specifically. This design sits at
that intersection.

---

## 3. Research Questions

### 3.0 Two questions, and why one is primary

**Q1 (primary). How does post-training quantization affect contamination-detection signals?**
Specifically: does quantization differentially modulate peakedness-based detection (CDD) versus
probability-based detection (perplexity, Min-k% Prob), to the point of changing which detector family
is more reliable at a given precision?

- **Q1a.** Does quantization shift per-item detector scores? (Paired comparison, same item scored at each
  precision.)
- **Q1b.** Does quantization change a detector's ability to separate contaminated from clean items (AUC),
  and does it change the *ranking* between detector families?

**Q2 (secondary). Is there a quantization × contamination interaction in pass@1?**
Does the accuracy drop from quantization differ in size between a contamination-suspect benchmark
(HumanEval) and a plausibly clean one (LiveCodeBench post-cutoff)?

Q2 was this design's original primary question; we demote it after computing that at the sample sizes
realistically available (in particular, HumanEval's fixed ceiling of 164 items), Q2's minimum detectable
effect is **15.5 percentage points** even with infinite clean-condition data (§4.5.3), which is larger
than any effect size the quantization-in-code literature (§2.7) would lead us to expect. Q1, in contrast,
is answerable at the same 164-item ceiling: Q1b can detect a 0.05 AUC difference at n=164 with paired
detector scores (§4.5.2), and Q1a — which does not depend on contamination labels at all — needs as few
as 87–196 items depending on effect size (§4.5.1). **Q1a is therefore the result this design is guaranteed
to be able to report; Q2 is best-effort.**

### 3.1 Q2 design: a 2×2 comparison

|  | Full precision | Quantized | Difference |
|---|---|---|---|
| **Contamination-suspect** (HumanEval) | A | B | A − B |
| **Clean** (LiveCodeBench post-cutoff) | C | D | C − D |

The quantity of interest is the interaction **(A − B) − (C − D)**, estimated on the **log-odds scale**
(§4.5.3 explains why raw percentage points are unsafe here) via the `precision:contaminated` term of a
mixed-effects logistic regression (§4.5.5).

- **Near zero:** quantization's accuracy drop is not modulated by contamination status. Existing papers'
  *drop* estimates may transfer across contamination status even though their *absolute* accuracy numbers
  remain potentially inflated by contamination — this distinction matters and we report both separately.
- **Positive:** the drop is larger under contamination — consistent with the hypothesis that quantization
  partially erases memorized answers, so some of the reported "capability loss" from quantization is
  actually loss of memorized content.
- **Negative:** the drop is smaller under contamination — consistent with arXiv:2410.16454's finding that
  quantization can *resurface* suppressed content rather than erase it.

A negative interaction does **not** mean absolute accuracy rises under contamination after quantization;
it means the *drop* is comparatively smaller (e.g., 0.85→0.83 vs. 0.35→0.31 is a −2pp interaction despite
both conditions declining). Testing arXiv:2410.16454's "resurfacing" claim specifically requires comparing
the quantized-contaminated cell against its own full-precision baseline (B vs. A) directly, not the
interaction term, and we report both.

### 3.1.1 The three outcomes of Q2 are not symmetric

The base-rate confound quantified in §4.5.3 produces a spurious interaction that is **always negative**
(−1 to −3pp across plausible model assumptions) even when the true effect is exactly zero. This means:

| Observed interaction | Relationship to the artifact | Evidentiary status |
|---|---|---|
| **Positive** | Opposite sign from the artifact | Conservative — hardest to explain away, most publishable as-is |
| **Negative** | Same sign as the artifact | Requires the mitigations in §4.5.3–§4.5.4 (log-odds scale, primary contrast restricted to LCB pre/post, difficulty-stratification check) before it can be trusted |
| **Null** | — | Only interpretable with a pre-specified equivalence margin (§4.5.3); an unmargined null is "we couldn't tell," not "there is no effect" |

Q2 is therefore not "any of three outcomes makes a paper" — only the positive branch is unconditionally
safe. This is the second reason Q1 carries the paper's primary claim: **Q1's outcome does not depend on
Q2's sign**, and is informative whether or not Q2 clears its bar.

### 3.2 Framing discipline

The introduction's motivating claim ("some of the reported quantization accuracy drop is actually lost
memorization") is the paper's *motivation*, tested indirectly and partially through Q2's sign and through
the direct B-vs-A comparison in §3.1. The paper's *contribution* claim, stated in the abstract and §1, is
Q1. We keep these separate throughout to avoid the failure mode of claiming to have measured something the
design cannot actually power.

---

## 4. Method

### 4.1 Models

| Model | Size | Baseline precision | Role | Notes |
|---|---|---|---|---|
| Qwen2.5-32B | 32.5B | fp16 | Primary | Dense, GQA+RoPE; no official QAT checkpoint exists, so naive-PTQ comparisons are uncontaminated by a QAT confound |
| Llama-3.3-70B | 70B | fp16 (**must be re-run**; only an int8 baseline currently exists) | Primary | arXiv:2505.20276 reports pronounced BNB-nf4 fragility (32% drop) in **Llama-3.1-70B**, not 3.3 — the closest available evidence, but the 3.1→3.3 version difference is itself untested (§6) and this is our best candidate for a large effect, not a confirmed one |
| Qwen2.5-7B | 7B | fp16 | Pilot workhorse + size axis | Cheap to run; used to secure item counts for the pilot and as a secondary "does the effect scale with model size" probe |
| Gemma-4-31B-it *(appendix only)* | ~30.7B | bf16 | **Excluded from main analysis** | Ships an official QAT q4_0 checkpoint (structurally different 4-bit degradation vs. naive PTQ), has a thinking-mode toggle, and is multimodal — three uncontrolled confounds. Retained only for an appendix "QAT vs. PTQ" comparison. |

Architecture is not treated as a controlled axis: Qwen2.5 and Llama-3.3 are both dense, GQA+RoPE
transformers, so this column would carry no information. The only remaining axis across models is size.
All within-model comparisons use the model's own full-precision baseline; we do not attempt to unify
baseline precision *across* models (fp16 vs. bf16), since Gemma's exclusion makes the remaining two model
families both fp16-comparable already.

### 4.2 Data: contamination axis

| Axis | Condition | Target *n* | Rationale | Note |
|---|---|---|---|---|
| Primary contamination | LiveCodeBench, pre-cutoff | ≥ 1,000 | Same source and format as the clean control; only publication date differs, minimizing the difficulty confound in §4.5.3 | Promoted from secondary status after §4.5.3's power analysis |
| Primary control | LiveCodeBench, post-cutoff | ≥ 1,000 | After the **latest** of the three models' training cutoffs | "Clean," subject to the caveats in §6 |
| Secondary contamination | HumanEval | 164 (hard ceiling) | Released 2021; plausible contamination for all three models | Sufficient for Q1 (§3.0) but not Q2 (§4.5.3) |
| Secondary contamination | MBPP+ | 378 | Separate arm | **Not pooled** with HumanEval — different difficulty distributions would reintroduce the base-rate confound *inside* a nominally single condition. The combined n=542 is a sample-size reference only, never a pooled analysis cell. |

Pre/post-cutoff is the primary axis for two reasons: it isolates contamination while holding
source/format/difficulty roughly constant (ruling out "it's a different kind of problem" as an
alternative explanation), and — because difficulty is held roughly constant — it also directly reduces
the base-rate confound quantified in §4.5.3. One design change addresses two problems.

HumanEval and MBPP+ are usable for **Q1** at their native sample sizes (§3.0) but are demoted to secondary
evidence for **Q2**, where the 164-item ceiling on HumanEval is the binding constraint on detectable
effect size (§4.5.3) regardless of how much clean-condition data is collected.

### 4.3 Quantization axis

fp16 / bf16 (baseline) → **BNB int8** → **BNB int4-nf4** → **GPTQ-int4 or AWQ-int4**

We do not include a double-quantization condition: double quantization affects memory footprint but not
accuracy at a level distinguishable from measurement noise, so it carries no information as an
experimental condition. The fourth level is GPTQ/AWQ instead, which serves two purposes:
it is required to make any "quantization in general" claim (arXiv:2505.20276 itself concludes effects
depend heavily on technique, model, and task), and — because the code-domain consensus in §2.7 (little to
no degradation) applies specifically to *calibration-based* methods on *code-specialized* models, while
arXiv:2505.20276's 32%-drop result is for *calibration-free* BNB-nf4 — the **bnb-nf4 arm is expected to
show the largest effect size** of the four conditions. The pilot (§4.7) runs this arm first.

### 4.4 Detection signals (for Q1)

- **Peakedness family:** CDD — *Contamination Detection via output Distribution* — the
  output-distribution-peakedness detector characterized in §2.4, introduced by Dong et al. (2024).
  (Expansion and attribution as given in arXiv:2603.03203's abstract and introduction.)
- **Probability family:** perplexity, Min-k% Prob.

**Sampling protocol** (following the CDD original paper, as replicated in arXiv:2603.03203): one greedy
(temperature 0) generation plus *n* temperature-0.8 samples per item, at each precision level. This must
be budgeted into the generation-cost estimate in §5 (step 2) alongside the continuous-scoring pipeline
(step 1), since both can share the same underlying generations.

**Threshold handling (ξ).** The original CDD paper fixes a detection threshold ξ=0.01, calibrated on 7B
models; arXiv:2603.03203 re-selects ξ per condition via Youden-index maximization on its own small models,
and explicitly notes this "gives CDD every advantage" — i.e., it is an optimistic, oracle-selected
threshold. Because **Q1b's primary metric is AUC, which is threshold-independent**, ξ recalibration is
*not* required for Q1b. It is only relevant if CDD point-accuracy is reported as a secondary descriptive
statistic, in which case ξ must **not** be re-selected on the evaluation set per condition (which would
reproduce the original paper's optimistic-oracle bias); instead, apply either an identical, pre-fixed
threshold across all precision conditions, or a threshold calibrated on a held-out split.

### 4.5 Statistical design

#### 4.5.1 Q1a — paired detector-score shift

Item-level, same-item comparison across precision; a paired t-test / mixed-effects equivalent.

| Effect size (Cohen's d) | Items needed (80% power, α=0.05, paired) |
|---|---|
| 0.3 | 87 |
| 0.2 | 196 |

**Caveat:** these d values are illustrative, not predictions. §2.4 establishes that CDD behaves as a step
function around a memorization threshold whose location for 32B–70B models is unknown; if quantization
moves a model across that threshold, d could be far larger or smaller than either row above. Probability-
based detectors are not reported to exhibit step-function behavior, so their d is expected to be more
stable — an additional reason (beyond §2.4's AUC argument) to treat them as the primary detector family
and CDD as a comparison arm. The pilot (§4.7) measures d directly before the full run is sized.

#### 4.5.2 Q1b — detector-family AUC comparison, and its dependence on label quality

AUC separability of contaminated vs. clean items, per detector, per precision; compared as **paired**
AUCs (both computed from the same item set) since the same items are scored by both detectors.

| Items per condition | SE(AUC) | Detectable ΔAUC, r=0 | r=0.8 | r=0.9 |
|---|---|---|---|---|
| 164 (HumanEval ceiling) | 0.029 | 0.114 | **0.051** | 0.036 |
| 300 | 0.021 | 0.084 | 0.038 | 0.027 |
| 542 (HumanEval+MBPP+, reference only) | 0.016 | 0.063 | 0.028 | 0.020 |
| 1,000 | 0.012 | 0.046 | 0.021 | 0.015 |

At r=0.8, the 164-item ceiling's detection limit is 0.051 — **just short** of a 0.05 target (solving
exactly gives 170 items, not 164). HumanEval alone is therefore a hair short of the target, and MBPP+ or
additional LCB items are needed to close the gap.

**Label noise is the more serious threat to Q1b.** Until TRACER (§5, step 5) provides a direct
contamination measurement, the pre/post-cutoff split is a **proxy** label with some error rate *e*. Label
noise does not primarily inflate standard error — it **attenuates the true AUC difference itself**,
roughly as ΔAUC_observed ≈ (1 − 2e) × ΔAUC_true:

| Proxy-label error rate *e* | Observed ΔAUC (true = 0.050) | Items needed (r=0.8) |
|---|---|---|
| 0% (proxy is exact) | 0.050 | 170 |
| 10% | 0.040 | 287 |
| 20% | 0.030 | 541 |
| 30% | 0.020 | 1,268 |

At e=20%, Q1b's item requirement rises to Q2's level (≈542); Q1b's power advantage over Q2 is therefore
conditional on label quality, which is why TRACER's residual-contamination measurement (§5, step 5) is a
prerequisite for Q1b specifically, not merely a nice-to-have for Q2. **Q1a is immune to this problem**,
since it never uses a contamination label — only within-item, cross-precision score comparisons. This is
the second reason Q1a, not Q1b, is the result this design is guaranteed to be able to report.

#### 4.5.3 Q2 — pass@1 interaction, base rate, and scale

**Unpaired baseline (p=0.5, most conservative), 4-cell difference-in-differences, α=0.05 two-sided:**

| Items per condition | Power @ 5pp | @ 10pp | @ 20pp |
|---|---|---|---|
| 50 | 0.07 | 0.11 | 0.29 |
| 164 (HumanEval ceiling) | 0.10 | 0.24 | 0.72 |
| 400 | 0.17 | 0.51 | 0.98 |
| 800 | 0.29 | 0.81 | 1.00 |
| 1,600 | 0.51 | 0.98 | 1.00 |
| 3,200 | 0.81 | 1.00 | 1.00 |

Items needed for 80% power: **≈196** (20pp effect), **≈785** (10pp), **≈3,140** (5pp).

**Sample-size decomposition (do not conflate these — they answer different questions):**

| Scenario | *n* needed (10pp) | Source of the reduction |
|---|---|---|
| Unpaired, p=0.5 both conditions | **785** | — (assumption-free upper bound; **use this for planning**) |
| Unpaired, actual base rates 0.85/0.35 | 557 | Base rate alone: −29% (extreme base rates shrink binomial variance) |
| Paired (item difficulty SD=1.5 model), p=0.5 | 549 (implied r≈0.30) | Pairing alone: −30% |
| Paired (SD=1.5 model), actual base rates | **≈415–419** | Both effects combined: −47% |

785 and ≈417 differ by nearly 2×; **785 is the number to plan against**, since it assumes nothing about
base rates or item-level correlation, both of which must be *measured*, not assumed. (Note that these
reductions do not compose: applying a base-rate-derived correlation estimate to the p=0.5 sample-size
formula mixes two different scales and is not a valid shortcut to the paired-and-base-rate-adjusted
figure.) The true item-level correlation between precisions may exceed the model's implied r≈0.29–0.30 — the same
prompt and decoding strategy is used for both precisions, so more is shared between conditions than
difficulty alone — and could plausibly reach 0.6–0.9, which would bring the requirement down to 157–314.
This is unverified and must be measured in the pilot (§4.7), not assumed in the plan.

**The base-rate confound.** HumanEval (fp16 pass@1 ≈ 0.85) and LiveCodeBench-post (≈0.35) have very
different baseline accuracies. On the raw percentage-point scale, this difference alone produces a
**spurious interaction** even when the true, item-conditional quantization effect (in log-odds) is
*identical* across both conditions:

| Item-conditional log-odds drop β | HumanEval drop | LCB-post drop | **Spurious %p interaction** |
|---|---|---|---|
| 0.25 | 2.6pp | 4.0pp | **−1.4pp** |
| 0.50 | 5.6pp | 7.7pp | **−2.2pp** |
| 0.75 | 8.8pp | 11.2pp | **−2.5pp** |
| 1.00 | 12.3pp | 14.5pp | **−2.2pp** |

The spurious effect is **always negative** across this range — i.e., it has the same sign as the
"quantization resurfaces memorization" hypothesis (§3.1) — meaning percentage-point analysis is at
meaningful risk of confirming that hypothesis for the wrong reason. If the true target effect is 5pp, this
artifact is up to half that size.

**Mitigation:** report the primary Q2 result as a **log-odds-scale interaction term**, not percentage
points. This does not eliminate confounding by assumption — it assumes the quantization effect is constant
on the *odds-ratio* scale, which can itself be wrong (e.g., if quantization disproportionately harms hard
items). **Difficulty stratification is therefore not an optional robustness check but a required
assumption-validation step**: bin items by difficulty and confirm β does not vary systematically across
bins before interpreting the log-odds interaction term. If it does vary, a difficulty-matched design is
needed instead. Even granting the log-odds assumption, a small residual bias remains: computing log-odds
from *aggregated* (marginal) accuracy rather than fitting the item-conditional mixed model leaves a
systematic bias of up to about 0.023 in the interaction term (verified both analytically and against an
independent numerical-integration check for this document, not merely a Monte Carlo artifact) — small
relative to a 5pp target effect (~0.2 in log-odds at p≈0.5, so the residual bias is at most ~11–12% of the
target) but roughly 4× smaller, proportionally, than the ~50% relative bias percentage-points would carry
for the same target effect (the spurious-interaction table above). This is an argument *for* the log-odds
scale, not a reason to abandon it — but it is only avoided
entirely by fitting the full item-conditional mixed-effects model (§4.5.5), not by computing log-odds from
cell-aggregate accuracies as a shortcut.

#### 4.5.4 Reconciling the numbers

Two honest numbers coexist: **785** is the assumption-free planning target. **≈417** is what the model in
§4.5.3 predicts *if* its assumptions (item difficulty SD=1.5, actual base rates, conditional independence
given difficulty) hold. Plan for 785; treat anything the pilot buys below that as upside, not as a
requirement.

#### 4.5.5 Statistical model

```
correct ~ precision * contaminated + (1 | item) + (1 | model)
```

Mixed-effects logistic regression; the `precision:contaminated` interaction term is the estimand of
interest for Q2, reported on the log-odds scale with a confidence interval. Item and model random effects
absorb both the pairing benefit (§4.5.3) and cross-model heterogeneity without requiring the analyst to
assume a value for the item-level correlation r in advance. (A two-sample test such as Welch's t-test is
not appropriate here, since the estimand is a 4-cell difference-in-differences with repeated measures on
items.)

### 4.6 Pilot gate: CDD baseline AUC

Because CDD's behavior is a step function around an unknown threshold (§2.4), Q1b's feasibility must be
checked empirically before committing to the full run. Using the 542-item, r=0.8 reference point:

| CDD baseline AUC (16-bit) | Quantization-induced ΔAUC (assumed 10% reduction in separation) | Detection limit | Detectable? |
|---|---|---|---|
| 0.52 (≈chance) | 0.002 | 0.031 | **No** |
| 0.60 | 0.010 | 0.030 | **No** |
| 0.70 | 0.019 | 0.028 | **No** |
| 0.85 | 0.026 | 0.021 | Yes |
| 0.95 | 0.019 | 0.012 | Yes |

The break-even point is a baseline AUC of **≈0.79**. The failure mode to plan for is not a ceiling effect
but a **floor** effect: if 32B–70B models sit near CDD's chance-level regime, ΔAUC ≈ 0 regardless of
sample size, and no amount of additional data recovers detectability. **Gate:** measure CDD's 16-bit
baseline AUC in the pilot; if it is below 0.79, drop CDD from the primary analysis and report Q1 using the
probability-based detectors only, with CDD's (in)ability to function at this scale reported as a
standalone finding (Contribution 4, §1).

### 4.7 Pilot study

**Qwen2.5-7B, BNB-nf4 arm first** (§4.3 — the arm expected to show the largest effect, so the pilot is
maximally informative about worst-case behavior).

Four quantities must be measured, together, before finalizing sample sizes — measuring only one leaves the
plan unable to locate itself within the tables in §4.5:

(a) Q1a detector-score shift size *d*; (b) Q1b's observed AUC and the cross-precision AUC correlation *r*;
(c) Q2's log-odds effect size and item-level correlation *r*; (d) each condition's actual base-rate
accuracy. If the CDD baseline AUC measured in (b) is below 0.6, prioritize completing the TRACER
residual-contamination measurement (§5, step 5) before proceeding, since a sub-0.6 AUC is itself a sign of
high label noise in the proxy contamination labels (§4.5.2).

---

## 5. Execution Plan

1. **Build the continuous-scoring pipeline first** (partial test-case pass rate + token log-probability).
   Per §4.5.3, no achievable item count rescues 0/1 pass@1 as the primary metric — this is a prerequisite,
   not step one of many equally-weighted steps.
2. **Build the detector-scoring pipeline** (CDD, perplexity, Min-k% Prob per item, per precision) —
   required for Q1. Budget CDD's per-item multi-sample requirement (§4.4) into the generation-cost
   estimate; design steps 1 and 2 to share underlying generations wherever possible.
3. **Count available LiveCodeBench pre-/post-cutoff items** against the latest model's actual cutoff.
   Target ≥1,000 each. If this target cannot be met, **demote Q2 to a secondary, confidence-interval-only
   analysis** — this does not block the project, since Q1 does not depend on it.
4. **Verify actual training cutoffs** via LLMLagBench (arXiv:2511.12116) rather than trusting declared
   dates.
5. **Measure residual contamination** in LCB pre/post and HumanEval via TRACER (arXiv:2605.24079). This is
   a prerequisite for Q1b specifically (§4.5.2) and can proceed in parallel with step 6, since Q1a does not
   require it.
6. **Pilot** (§4.7): Qwen2.5-7B, BNB-nf4 arm first. Precondition: measure the CDD baseline AUC gate (§4.6)
   before interpreting any Q1b pilot numbers; if the gate fails, switch the primary detector to
   probability-based methods for the remainder of the pilot.
7. **Recompute power** from pilot-measured effect sizes and correlations (§4.5.4). If Q2's required *n*
   is unreachable, keep it as a confidence-interval-only secondary result rather than forcing significance.
8. **Full run.** Store item-level raw data for every condition: pass@1, partial credit, token
   log-probability, and all three detector scores. Aggregate-only storage would foreclose the paired and
   mixed-effects analyses this design depends on.
9. **Analysis.**
   - *Q1a:* paired, item-level comparison of detector scores across precision (mixed-effects, item random
     effect).
   - *Q1b:* per-precision AUC with paired-AUC confidence intervals; report any detector-ranking reversal
     explicitly.
   - *Q2:* `correct ~ precision * contaminated + (1|item) + (1|model)`, log-odds interaction term and CI;
     interpret only after the difficulty-stratification check (§4.5.3) confirms the constant-odds-ratio
     assumption.

---

## 6. Threats to Validity

- **Observational, not causal.** Contamination status cannot be randomly assigned to off-the-shelf,
  already-pretrained 7B–70B models; it is an observed covariate, not a treatment. Results are reported as
  associations, following arXiv:2501.18771's causally-identified design as an aspirational reference this
  study cannot replicate at this model scale (§2.3).
- **Declared training-cutoff dates may be wrong.** Mitigated via LLMLagBench verification (§5, step 4);
  even so, residual uncertainty in cutoff dates should be reported as a limitation on the pre/post-cutoff
  split's precision.
- **"Filtered by date" does not guarantee "uncontaminated."** arXiv:2602.12413 and arXiv:2311.04850
  document that semantic duplication and paraphrase evade time- and n-gram-based filtering. We do not
  claim LiveCodeBench-post is a clean ground truth; we describe it as "lower-contamination" and use
  TRACER (§5, step 5) to measure, rather than assume, residual contamination in every condition, feeding
  that measurement into the label-noise correction in §4.5.2.
- **Extrapolation risk from arXiv:2603.03203.** That paper's findings are established at 70M–410M, roughly
  one to three orders of magnitude below this design's 7B–70B range, and the paper itself disclaims
  extrapolation. We treat CDD's operability at this scale as an open empirical question gated by pilot
  measurement (§4.6), not as an assumption, and report the answer (whichever direction it goes) as a
  standalone contribution.
- **Version mismatch in the bnb-nf4 effect-size prior.** The 32% BNB-nf4 drop cited to motivate treating
  Llama-3.3-70B as the largest-expected-effect arm (§4.1, §4.3) was measured in arXiv:2505.20276 on
  **Llama-3.1-70B**. Llama-3.3-70B is a later, instruction-tuned-differently release in the same nominal
  family; whether it shares 3.1's specific BNB-nf4 fragility has not been established and should be
  treated as an open question the pilot (§4.7) can speak to, not an assumption carried into the main
  analysis.
- **Gemma-4-31B-it's confounds** (official QAT checkpoint, thinking-mode toggle, multimodality) make it
  unsuitable for the naive-PTQ comparison this design otherwise controls for. It is excluded from the
  primary analysis and used only in an appendix QAT-vs-PTQ comparison.
- **Base-rate confound between conditions** (§4.5.3) is mitigated on the log-odds scale under a
  constant-odds-ratio assumption, which is itself validated (not assumed) via difficulty stratification.
  If that assumption fails, the design falls back to a difficulty-matched comparison rather than reporting
  an uninterpretable interaction term.
- **Proxy contamination labels carry error** until TRACER measurement is complete (§4.5.2); this
  attenuates, rather than adds noise to, the true effect being measured in Q1b, and is addressed by
  treating TRACER measurement as a Q1b prerequisite rather than an optional check.
- **Scope limits.** Three models, four quantization configurations, code generation only, predominantly
  Python. Findings need not generalize to other architectures (e.g., MoE), other tokenizers, or other
  domains (e.g., natural-language QA), and we do not claim they do.

---

## 7. Expected Contributions

Restated from §1 with their evidentiary basis:

1. **Primary (Q1):** the first measurement of quantization's effect on contamination-detection signal
   families at 7B–70B scale — either confirming that probability-based detectors remain reliable while
   peakedness-based detection is disrupted, or the reverse, or no differential effect. All three outcomes
   are informative and reportable (§3.1.1's asymmetry argument applies to Q2, not Q1).
2. **Secondary (Q2):** a bounded, log-odds-scale, base-rate-corrected estimate (or confidence interval, if
   underpowered) of the quantization × contamination interaction in code-generation pass@1, explicitly
   framed as associational.
3. **Byproduct:** a released item-level dataset (pass@1, partial credit, log-probability, three detector
   scores × 4 quantization levels × 3 models × 4+ benchmark conditions) intended to outlive this paper's
   specific conclusions and support future contamination-detection benchmarking.
4. **Boundary result:** direct evidence on whether CDD (or contamination detection generally) is operative
   at 32B–70B scale — a gap explicitly left open by arXiv:2603.03203 — regardless of which way Q1 comes
   out.

## 8. Limitations

Stated here upfront, in the spirit of a registered report, rather than deferred to a post-hoc discussion
section:

- No causal claims are possible without random assignment of contamination, which cannot be done on
  off-the-shelf pretrained models (§6).
- Q2 may remain underpowered even after all mitigations in §4.5; if so, it is reported as a
  confidence-interval-bounded secondary result, not as a significance claim, and the paper's contribution
  claim does not depend on it clearing significance.
- CDD may be entirely inoperative at 32B–70B scale (the "floor" failure mode, §4.6); if the pilot gate
  fails, this is reported as Contribution 4, not treated as a design failure requiring a redesign.
- The scope is limited to three code-generation-capable dense transformer models and four quantization
  configurations; generalization beyond this scope is not claimed.

---

## References

Numbered by arXiv ID.

**Contamination — surveys**
- arXiv:2404.00699 — *A Comprehensive Survey of Contamination Detection Methods in Large Language Models* (TMLR 2025)
- arXiv:2502.14425 — *A Survey on Data Contamination for LLMs*
- arXiv:2502.17521 — *Recent Advances in Large Langauge Model Benchmarks against Data Contamination: From Static to Dynamic Evaluation* [sic, original typo preserved]
- arXiv:2605.26133 — *Pretraining Data Exposure: Survey of Membership Inference*

**Temporal-split contamination measurement**
- arXiv:2310.10628 — *Data Contamination Through the Lens of Time*
- arXiv:2403.07974 — *LiveCodeBench*
- arXiv:2511.12116 — *LLMLagBench: Identifying Temporal Training Boundaries*
- arXiv:2504.14655 — *LeetCodeDataset: Temporal Dataset for Robust Evaluation*

**Effect sizes of contamination**
- arXiv:2501.18771 — *Overestimation in LLM Evaluation* (controlled, machine translation)
- arXiv:2403.04811 — *Quantifying Contamination in Code Generation Evaluation*
- arXiv:2506.02791 — *Rethinking the Effects of Data Contamination in Code Intelligence*
- arXiv:2507.19219 — *How Much Do LLMs Cheat? One-Time-Pad Framework*

**Limits of contamination detection**
- arXiv:2311.04850 — *Rethinking Benchmark and Contamination with Rephrased Samples* (title may be abbreviated; confirm the full official title against arXiv before submission)
- arXiv:2602.12413 — *Soft Contamination Means Benchmarks Test Shallow Generalization*
- arXiv:2402.02823 — *Evading Data Contamination Detection is (too) Easy*
- arXiv:2409.09927 — *Limitations, Inconsistencies, and Oracle Challenges* (title appears truncated in this project's source list; confirm against arXiv before submission)
- arXiv:2603.03203 — *No Memorization, No Detection: Output Distribution-Based Contamination Detection in Small Language Models*

**Code-benchmark contamination**
- arXiv:2605.24079 — *TRACER: Semantic-Aware Fine-Grained Code Contamination Detection*
- arXiv:2411.10842 — *CodeCleaner: Contamination Mitigation Toolkit*
- arXiv:2503.06643 — *Is Your Benchmark Still Useful? Dynamic Benchmarking for Code*
- arXiv:2503.13572 — *VeriContaminated: LLM-Driven Verilog Coding*

**Post-hoc decontamination**
- arXiv:2509.15218 — *LNE-Blocking: Contamination Mitigation Evaluation*
- arXiv:2601.19334 — *When Benchmarks Leak: Inference-Time Decontamination*
- arXiv:2506.04142 — *Trustworthy LLM Evaluation via Shortcut Neuron Analysis*
- arXiv:2605.21543 — *Provable Joint Decontamination for Multiple LLMs*

**Quantization**
- arXiv:2503.07103 — *Evaluating the Impact of Post-Training Quantization on Large Language Models for Code Generation*
- arXiv:2507.09665 — *Is Quantization a Deal-breaker? Empirical Insights from Large Code Models*
- arXiv:2506.22776 — *Smaller = Weaker? Benchmarking Robustness of Quantized LLMs in Code Generation*
- arXiv:2505.20276 — long-context quantization evaluation (title not independently verified in source documents; cited only for its BNB-nf4 / Llama-3.1-70B effect size, §2.7, §4.3)

**Quantization × unlearning/memorization**
- arXiv:2410.16454 — quantization reverses machine unlearning, 21%→83% retained knowledge after 4-bit quantization (title not independently verified; cited only for the 21%→83% figure, §1–§2)
- arXiv:2605.15138 — *Forgetting That Sticks: Quantization-Permanent Unlearning via Circuit Attribution*
