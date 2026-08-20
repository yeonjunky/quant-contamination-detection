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
retained for utility-constrained unlearning methods in one study; the mechanism has been formalized as a sparsity-permanence tradeoff in which
sub-threshold parameter changes are erased by quantization's bin structure). Separately, contamination-detection
research shows that detectors are not interchangeable: output-distribution-peakedness detectors (CDD)
require verbatim memorization and collapse to chance without it, while probability-based detectors
(perplexity, Min-k% Prob) remain informative under the same conditions. No published work has examined
what happens at the intersection: does post-training quantization differentially reshape these two
families of contamination signal, and does it change the outcome of contamination-vs-clean comparisons
on code-generation benchmarks?

We present a registered-report-style observational design (analysis plan fixed before execution, though
not lodged with an external registry) to answer this at a scale (7B–32B) roughly one to two and a half
orders of magnitude larger (17×–464×, depending on which endpoints are compared) than the only prior
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
resurfaces after quantization: one study reports retained-knowledge rates for utility-constrained
unlearning methods rising from 21% at full precision to 83% after 4-bit quantization
(arXiv:2410.16454; the qualifier is the source's own — it warns that figures from unconstrained
methods are misleading). A mechanistic follow-up explains why:
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
at the 32B scale either. What the paper does establish is a *mechanism*: CDD is governed by a sharp
memorization threshold tied to the absolute number of trainable parameters, not model size per se.

Put together, these two threads motivate a question that neither literature has asked: **if quantization
measurably disturbs memorized traces, and contamination detectors differ in how much they depend on those
traces, does quantization differentially reshape what different contamination detectors see?**

Contamination assessments made on full-precision checkpoints may not transfer to quantized deployments
if detector scores shift with precision. Quantization is a common deployment transformation, and its grid
structure can disturb memorized traces.

A related but harder-to-power question — whether the
*quantization-induced accuracy drop itself* differs between contaminated and clean benchmark conditions —
is under-powered at benchmark-imposed sample sizes and is therefore **secondary** (§3.0, §4.5).

**Contributions (stated as design commitments, to be confirmed or refuted by execution):**

1. The first measurement, at 7B–32B scale, of how post-training quantization affects the relative
   behavior of peakedness-based vs. probability-based contamination-detection signals (Q1a/Q1b).
2. A bounded, log-odds-scale estimate of the quantization × contamination interaction on pass@1 in code
   generation, reported as an association rather than a causal effect, with an explicit base-rate-confound
   correction (Q2).
3. A reusable item-level dataset — pass@1, partial credit, token log-probability, and three detector
   scores, crossed with quantization technique, precision, and model — intended to support future
   contamination-detection benchmarking work independent of this paper's own conclusions.
4. Evidence on whether CDD is operative at 32B scale, which arXiv:2603.03203 did not test.

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
arXiv:2504.14655 (*LeetCodeDataset*) applies the same temporal-split principle to LeetCode problems; we
use LiveCodeBench for its larger dated pool, but the existence of an independent second instance
corroborates the design pattern.
Because publicly declared training-cutoff dates can be wrong or absent, we additionally rely on
arXiv:2511.12116 (*LLMLagBench*), which estimates a model's *actual* temporal training boundary from its
knowledge of recent events, as a validation step (§5, step 4) rather than trusting declared cutoffs
outright.

### 2.3 Effect sizes of contamination
arXiv:2501.18771 provides a controlled, causally-identified estimate of contamination's effect by directly
pretraining 1B/8B models on machine-translation data with contamination injected at controlled stages,
scales, and formats. We treat this as the methodological reference point for causal identification, but
**cannot replicate its design**: it requires pretraining from scratch, and our study uses off-the-shelf
7B–32B models in the code domain. We can therefore only approximate its causal design observationally
(§6). arXiv:2403.04811 and arXiv:2506.02791 quantify contamination's effect size specifically in code
generation, with the latter noting that most prior work measures only sample-level contamination and
under-counts the more common partial-contamination case. arXiv:2403.04811 is additionally a
*methodological* source for this design, not only an effect-size one: its surface-plus-AST matching
pipeline, developed for HumanEval and MBPP against pretraining-scale corpora, is what we adopt for the
Olmo3 ground-truth labelling in §5, step 5. arXiv:2507.19219 offers a one-time-pad-based
framework for quantifying benchmark-score overestimation generally.

### 2.4 Limits of contamination detection
This subsection is the paper's most load-bearing prior work. arXiv:2311.04850 shows n-gram-based
decontamination filtering is trivially evaded by paraphrase or translation. arXiv:2602.12413 (*Soft
Contamination Means Benchmarks Test Shallow Generalization*) extends this: semantic (non-lexical)
duplication is undetectable by n-gram matching and was found pervasively in the Olmo3 pretraining corpus,
including in CodeForces-derived data (78% semantic duplication reported) — this is the strongest available
objection to treating any time-filtered benchmark as "clean," and we address it directly in §6. Because
Olmo3 is one of the models in our own design (§4.1), this result is not borrowed evidence about some other
model's corpus but a direct prior on the contamination labels of one of our arms — and the same corpus
openness that made the finding possible is what lets us measure, rather than assume, that arm's label
error rate (§4.5.2).
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
- **What it does not show:** anything about 32B models, or about pretraining-time (as opposed to
  fine-tuning-injected) contamination. The paper explicitly states its small-model findings "should not be
  extrapolated to larger scales without further investigation." It is tempting to argue in the opposite
  direction — that CDD's positive 7B result means CDD "works" at the scales this design targets — but that
  argument violates the same caveat it would have to cite. **The position of 32B models relative to CDD's memorization
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
benchmarking multiple models (directly relevant to our five-model comparison, §4.1); arXiv:2506.04142
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
is answerable at the same 164-item ceiling: Q1b can detect a 0.051 AUC difference at n=164 with paired
detector scores (detecting exactly 0.05 requires 170 items; §4.5.2), and Q1a — which does not depend on contamination labels at all — needs as few
as 87–196 items depending on effect size (§4.5.1). **Q1a is primary; Q2 is secondary.**

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

### 3.1.1 Interpretation of Q2

The base-rate confound quantified in §4.5.3 produces a spurious interaction that is **always negative**
(−1 to −3pp across plausible model assumptions) even when the true effect is exactly zero. This means:

| Observed interaction | Relationship to the artifact | Evidentiary status |
|---|---|---|
| **Positive** | Opposite sign from the artifact | Less affected by this artifact |
| **Negative** | Same sign as the artifact | Requires the mitigations in §4.5.3–§4.5.4 (log-odds scale, primary contrast restricted to LCB pre/post, difficulty-stratification check) before it can be trusted |
| **Null** | — | Only interpretable with a pre-specified equivalence margin (§4.5.3); an unmargined null is "we couldn't tell," not "there is no effect" |

Q1 does not depend on Q2's sign or significance.

### 3.2 Scope of claims

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
| Qwen2.5-32B-Instruct | 32.5B | bf16 | Primary | Dense, GQA+RoPE; no official QAT checkpoint exists, so naive-PTQ comparisons are uncontaminated by a QAT confound |
| Qwen2.5-7B-Instruct | 7B | bf16 | Pilot workhorse + size axis | Cheap to run; used to secure item counts for the pilot and as a secondary "does the effect scale with model size" probe |
| Llama-3.1-8B-Instruct | 8B | bf16 | Size axis + externally verified cutoff | The only main-analysis model whose training cutoff is already independently verified by LLMLagBench (arXiv:2511.12116): declared 2023-12, detected knowledge-drop changepoint 2023-03. We use the declared (later) date as the conservative contamination boundary; the detected/declared gap and its consequence for this arm's LCB pre-cutoff pool are discussed in §4.2. Family tie to arXiv:2505.20276's BNB-nf4 fragility prior, which was measured on Llama-3.1-**70B** — same family and data recipe, roughly 9× smaller, so a weak prior rather than a confirmed expectation (§6) |
| Olmo3-7B-Instruct | 7B | bf16 | Ground-truth label validation + size axis | Fully open training data across all stages (pretraining and post-training), enabling direct measurement (rather than assumption) of the proxy-label error rate *e* in §4.5.2. Dense transformer, so no architecture confound is introduced. Joins the pilot (§4.7) |
| Olmo3.1-32B-Instruct | 32B | bf16 | Ground-truth label validation + size axis | Official 32B final Instruct release (`allenai/Olmo-3.1-32B-Instruct`), replacing the unavailable Olmo3-32B-Instruct name in the original design and matching Qwen2.5-32B-Instruct's footprint |

All five arms use the instruction-tuned (\*-Instruct) releases: code-generation pass@1 under
instruction prompts is the measured quantity, the illustrative base rates in §4.5.3 are
instruction-tuned figures, and the LLMLagBench verification (§5, step 4) probes instruct checkpoints.
Shorthand names elsewhere in this document (e.g., "Qwen2.5-7B") refer to these Instruct checkpoints.
An instruct checkpoint also widens the contamination surface — benchmark items can enter through
post-training (instruction-tuning) data as well as pretraining — which is why the Olmo3 ground-truth
search in §5, step 5 covers both stages.

QAT-shipped models (e.g., Gemma-family official QAT checkpoints) are excluded entirely: an official QAT
checkpoint degrades structurally differently from naive PTQ, and the available QAT models add further
uncontrolled confounds (thinking-mode toggles, multimodality). A QAT-vs-PTQ comparison on such a model
is deferred to future work; the format of the shipped checkpoints (llama.cpp q4_0) would also require a
second inference stack, whose numerics differences would confound the very contrast of interest.

Architecture is still not treated as a controlled axis: Qwen2.5, Llama-3.1, and Olmo3 are all dense
transformers, so this column would carry no information. Two axes do vary deliberately across models:
**size**, and **training-corpus transparency**. The latter is introduced specifically to serve §4.5.2 —
Olmo3 releases its pretraining corpus, so contamination labels for its arm can be measured against the
training data directly instead of inferred from a release-date proxy. All within-model comparisons use the
model's own full-precision baseline, and all five models share the same baseline precision (bf16), so no
cross-model baseline-precision issue arises.

**Compute footprint.** The available hardware is a single H100 (80 GB), with a single H200 (141 GB)
obtainable on request. Weight footprints, before KV cache and activations:

| Model | bf16 | int8 | int4 (nf4) | Fits a single device at bf16? |
|---|---|---|---|---|
| Qwen2.5-7B / Olmo3-7B / Llama-3.1-8B | ~14–16 GB | ~7–8 GB | ~4–5 GB | Yes (H100) |
| Qwen2.5-32B / Olmo3.1-32B | ~64–65 GB | ~32 GB | ~18 GB | Yes, tightly (H100, ~15 GB left for KV cache); comfortably on the H200 |

Every arm therefore runs its complete quantization ladder — bf16 baseline included — on a single
available device. No arm requires a baseline at a different precision from any other, which keeps
Q1a's bf16-anchored within-model contrast directly comparable across all five models. Models above
32.5B are excluded from the design as a hard compute constraint (a single device cannot hold a 70B-class
bf16 baseline), and this scale ceiling is recorded as a scope limitation in §8.

### 4.2 Data: contamination axis

| Axis | Condition | Target *n* | Rationale | Note |
|---|---|---|---|---|
| Primary contamination | LiveCodeBench, pre-cutoff | 873 available | Same source and format as the clean control; only publication date differs, minimizing the difficulty confound in §4.5.3 | The pre-specified ≥1,000 target is not met |
| Primary control | LiveCodeBench, post-cutoff | 182 available | On or after 2025-01-01, the first day after the latest model-level cutoff | The pre-specified ≥1,000 target is not met; Q2 is secondary and confidence-interval-only |
| Secondary contamination | HumanEval | 164 (hard ceiling) | Released 2021; plausible contamination for all five models | Sufficient for Q1 (§3.0) but not Q2 (§4.5.3) |
| Secondary contamination | MBPP+ | 378 | Separate arm | **Not pooled** with HumanEval — different difficulty distributions would reintroduce the base-rate confound *inside* a nominally single condition. The combined n=542 is a sample-size reference only, never a pooled analysis cell. |

Pre/post-cutoff is the primary axis for two reasons: it isolates contamination while holding
source/format/difficulty roughly constant (ruling out "it's a different kind of problem" as an
alternative explanation), and — because difficulty is held roughly constant — it also directly reduces
the base-rate confound quantified in §4.5.3. One design change addresses two problems.

The common boundary is 2025-01-01. Under LiveCodeBench `release_v6`, it yields 873 pre-cutoff and 182
post-cutoff items (1,055 total). This data-availability check fails the pre-specified ≥1,000 target in
both cells, so Q2 is demoted to a secondary, confidence-interval-only analysis as specified below.

**Pre-specified boundary rule.** Cutoff evidence quality differs by arm, so the boundary is defined to
remain valid under the weakest evidence rather than under the most optimistic reading:

| Arm | Cutoff evidence | Evidence tier | Conservative exposure bound |
|---|---|---|---|
| Olmo3-7B / Olmo3.1-32B | Official model cards for both final Instruct checkpoints state `Date cutoff: Dec. 2024` ([7B](https://huggingface.co/allenai/Olmo-3-7B-Instruct), [32B](https://huggingface.co/allenai/Olmo-3.1-32B-Instruct)) | Official model-level declaration | 2024-12; month-level conservatism makes 2025-01-01 the first eligible post-cutoff day |
| Llama-3.1-8B | Declared 2023-12; externally verified by LLMLagBench (detected drop 2023-03) | Verified declaration | 2023-12 (declared) |
| Qwen2.5-7B / Qwen2.5-32B | No unambiguous official cutoff declaration | Release-date bound (weakest) | 2024-09-19 ([official release announcement](https://qwenlm.github.io/blog/qwen2.5/)); at day-level resolution, LCB-post eligibility begins 2024-09-20 |

A release date is an unconditionally valid upper bound — a model cannot have trained on data published
after its own release — so an arm with no declaration still has a defensible boundary. The LCB
post-cutoff boundary is pre-specified as **the latest of the per-arm conservative bounds**. The Olmo
model-level declaration is the latest: because it has month resolution, all of December 2024 remains
potentially exposed and LCB-post begins on 2025-01-01. Qwen2.5 retains its arm-level 2024-09-19 release
bound (LCB-post begins 2024-09-20), but it does not determine the common boundary.

One asymmetry specific to the Llama-3.1-8B arm: LLMLagBench detects its knowledge-drop changepoint at
2023-03, nine months before the declared 2023-12 cutoff (§4.1). We use the declared, later date as the
conservative boundary for what the model *may have seen* — a knowledge drop is evidence of thin coverage,
not proof of non-exposure. But if the effective pretraining boundary is in fact earlier, this arm's LCB
pre-cutoff pool (which begins 2023-05) contains items the model plausibly never saw, diluting its
contamination-suspect condition toward the clean side. Its secondary suspect conditions (HumanEval,
MBPP+, both released 2021) are unaffected.

Rather than merely flagging this, we pre-specify a **boundary sensitivity analysis**. Every analysis
that consumes this arm's LCB contamination labels (Q1b, Q2) is run twice: once under the **declared
boundary** (2023-12; the 2023-05–2023-12 window counts as contamination-suspect) and once under the
**detected boundary** (2023-03; the window counts as clean — which empties this arm's LCB-pre set,
since LCB collection begins 2023-05, and drops the arm from the LCB contrast, leaving HumanEval and
MBPP+ as its only suspect conditions). The declared-boundary run is primary; the detected-boundary run
is sensitivity. **Which run is which is fixed here, in advance, and is not revisited after seeing
results.** Agreement between the runs demonstrates robustness to the nine-month ambiguity; disagreement
quantifies exactly how much rides on it — either way a reportable outcome. The re-run costs no
additional generation or scoring: only analysis-time label assignment changes, and the item-level
raw-data requirement (§5, step 8) exists precisely to make such re-analyses possible.

The rule generalizes beyond this arm: **any arm whose cutoff evidence is bracketed by competing bounds is
re-analyzed under those bounds**. Qwen2.5 currently has only its release-date bound. Olmo3 has matching
model-card declarations and needs no sensitivity run. As a purely descriptive check, we also compare the ambiguous-window items'
full-precision detector-score distribution against the definitely-clean (post-boundary) and
definitely-suspect distributions; this comparison is reported separately and is **never fed back into
Q1b's labels** — doing so would let the detectors under evaluation adjudicate their own ground truth.

HumanEval and MBPP+ are usable for **Q1** at their native sample sizes (§3.0) but are demoted to secondary
evidence for **Q2**, where the 164-item ceiling on HumanEval is the binding constraint on detectable
effect size (§4.5.3) regardless of how much clean-condition data is collected.

### 4.3 Quantization axis

**bf16 baseline** → **BNB int8** → **BNB int4-nf4** → **GPTQ-int4 or AWQ-int4**

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

**The two detector families do not cost the same.** CDD requires the multi-sample generation above,
repeated at every precision level for every model. Perplexity and Min-k% Prob require no generation at
all: both are computed from teacher-forced log-probabilities over fixed text, i.e. a single forward pass
per item per precision. Generation cost is therefore borne almost entirely by CDD. If measured throughput
(§5, step 6) forces a reduction in scope, the only large saving available is CDD's sample count *n*, and
the resulting precision loss is confined to the CDD arm of Q1a and Q1b — the probability-based detector
results, which §2.4 makes the primary family, are unaffected. This asymmetry should be stated explicitly
whenever *n* is reduced, so that a budget decision is not mistaken for a finding about CDD.

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
function around a memorization threshold whose location for 32B models is unknown; if quantization
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
prerequisite for Q1b. Q1a does not use a contamination label and remains interpretable under label noise.

**The Olmo3 arm makes *e* measurable rather than assumed.** For every other model, *e* is unobservable:
the training corpora are closed, so the table above can only be read as a sensitivity analysis. Olmo3
releases its pretraining corpus, so benchmark items can be searched against the training data directly
(§5, step 5) and the pre/post-cutoff proxy can be scored against that result — yielding a measured *e* for
that arm instead of an assumed one. This does not change any figure in the table; it changes which column
we know we are in. It also supplies the one thing the label-noise correction otherwise lacks: an empirical
anchor for how good the release-date proxy actually is on this benchmark family. The prior from §2.4 is
not encouraging — arXiv:2602.12413 reports 78% semantic duplication in CodeForces-derived data within this
very corpus — which is precisely why measuring rather than assuming *e* matters here.

#### 4.5.3 Q2 — pass@1 interaction, base rate, and scale

**Unpaired baseline (p=0.5, most conservative), 4-cell difference-in-differences, α=0.05 two-sided:**

| Items per condition | Power @ 5pp | @ 10pp | @ 20pp |
|---|---|---|---|
| 50 | 0.06 | 0.11 | 0.29 |
| 164 (HumanEval ceiling) | 0.10 | 0.25 | 0.73 |
| 400 | 0.17 | 0.52 | 0.98 |
| 800 | 0.29 | 0.81 | 1.00 |
| 1,600 | 0.52 | 0.98 | 1.00 |
| 3,200 | 0.81 | 1.00 | 1.00 |

*Every cell is computed from the single formula power = Φ(δ/SE − z_{α/2}) + Φ(−δ/SE − z_{α/2}) with
SE = √(4·p(1−p)/n) at p = 0.5 — stated so that no cell can silently mix conventions.*

Items needed for 80% power: **≈196** (20pp effect), **≈785** (10pp), **≈3,140** (5pp).

**Sample-size decomposition (do not conflate these — they answer different questions):**

| Scenario | *n* needed (10pp) | Source of the reduction |
|---|---|---|
| Unpaired, p=0.5 both conditions | **785** | — (assumption-free upper bound; **use this for planning**) |
| Unpaired, actual base rates 0.85/0.35 | 557 | Base rate alone: −29% (extreme base rates shrink binomial variance) |
| Paired (item difficulty SD=1.5 model), p=0.5 | 555 (implied r = 0.293) | Pairing alone: −29% |
| Paired (SD=1.5 model), actual base rates | **≈415–419** | Both effects combined: −47% |

*Rows are computed under different conventions and must not be read as one continuum: row 2 evaluates
binomial variance at the null (no-drop) base rates; row 3 applies the σ=1.5 item model's implied
cross-precision correlation (r = 0.293, by numerical integration) to the row-1 figure. That rows 2 and 3
land within two items of each other (557 vs. 555) is coincidence — different mechanisms — and they do
not compose multiplicatively; row 4 is a separate joint computation.*

785 and ≈417 differ by nearly 2×; **785 is the number to plan against**, since it assumes nothing about
base rates or item-level correlation, both of which must be *measured*, not assumed. (Note that these
reductions do not compose: applying a base-rate-derived correlation estimate to the p=0.5 sample-size
formula mixes two different scales and is not a valid shortcut to the paired-and-base-rate-adjusted
figure.) The true item-level correlation between precisions may exceed the model's implied r ≈ 0.293 — the same
prompt and decoding strategy is used for both precisions, so more is shared between conditions than
difficulty alone — and could plausibly reach 0.6–0.9, which would bring the requirement down to ≈79–314.
This is unverified and must be measured in the pilot (§4.7), not assumed in the plan.

**The base-rate confound.** HumanEval (bf16 pass@1 ≈ 0.85) and LiveCodeBench-post (≈0.35) have very
different baseline accuracies. These two figures are illustrative values for a Qwen-class instruction-tuned
model; the actual base rates differ by model — Olmo3's in particular should not be assumed to match
Qwen2.5's — and are measured per model in the pilot (§4.7, item d). The argument below does not depend on
the specific values, only on the two conditions being far apart, which holds for every model in §4.1.
On the raw percentage-point scale, this difference alone produces a
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

#### 4.5.6 Multiplicity and confirmatory scope

The design generates many tests — three detectors × several precision contrasts × Q1a/Q1b × five
models — and declaring all of them at α=0.05 would make some spuriously significant results
near-certain. We therefore pre-specify a small **confirmatory family** and demote everything else to
exploratory status:

- **C1–C3 (Q1a):** the paired bf16 → BNB-nf4 detector-score shift on the LCB conditions, one test per
  detector (perplexity, Min-k% Prob, CDD) — the largest-expected-effect arm (§4.3) on the primary
  condition axis (§4.2).
- **C4 (Q1b):** whether the detector-family AUC ranking (probability-based vs. peakedness-based)
  differs between bf16 and BNB-nf4.

Holm correction is applied within this four-test family. Everything else — other quantization levels,
other models' arms, the HumanEval/MBPP+ secondary conditions, Q2 in its entirety, and the boundary
sensitivity re-runs of §4.2 — is reported as exploratory estimates with confidence intervals and exact
p-values, without significance claims. Powering the confirmatory family at the Holm-adjusted worst case
(α/4 for the smallest p-value; multiplier 3.339 in place of 2.802) raises Q1a's item requirement from
87 to ≈124 at d=0.3 — still within HumanEval's 164-item ceiling — and from 196 to ≈279 at d=0.2, which
requires the LCB conditions. The pilot-based power recomputation (§5, step 7) sizes the confirmatory
tests at these adjusted thresholds; the unadjusted tables in §4.5.1–§4.5.3 remain valid for the
exploratory analyses they describe.

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
but a **floor** effect: if 32B models sit near CDD's chance-level regime, ΔAUC ≈ 0 regardless of
sample size, and no amount of additional data recovers detectability. **Gate:** measure CDD's 16-bit
baseline AUC in the pilot; if it is below 0.79, drop CDD from the primary analysis and report Q1 using the
probability-based detectors only, with CDD's (in)ability to function at this scale reported as a
standalone finding (Contribution 4, §1).

### 4.7 Pilot study

**Qwen2.5-7B and Olmo3-7B, BNB-nf4 arm first** (§4.3 — the arm expected to show the largest effect, so the
pilot is maximally informative about worst-case behavior). Olmo3-7B is included in the pilot rather than
held for the main run because it is the only arm that yields a measured proxy-label error rate *e*
(§4.5.2), and *e* is a prerequisite for Q1b: discovering a large *e* after the full run is sized would
invalidate the sizing.

Five quantities must be measured, together, before finalizing sample sizes — measuring only one leaves the
plan unable to locate itself within the tables in §4.5:

(a) Q1a detector-score shift size *d*; (b) Q1b's observed AUC and the cross-precision AUC correlation *r*;
(c) Q2's log-odds effect size and item-level correlation *r*; (d) each condition's actual base-rate
accuracy, **per model** (§4.5.3); (e) the proxy-label error rate *e* on the Olmo3 arm, from the
corpus-search labels of §5, step 5, which locates the whole design within the label-noise table in §4.5.2.
If the CDD baseline AUC measured in (b) is below 0.6, prioritize completing the TRACER
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
3. **Count available LiveCodeBench pre-/post-cutoff items** against the common 2025-01-01 boundary.
   `release_v6` contains pre 873 / post 182 / total 1,055. The ≥1,000 target is therefore unmet in both
   cells, so **Q2 is a secondary, confidence-interval-only analysis**; Q1 remains unaffected.
4. **Verify actual training cutoffs** via LLMLagBench (arXiv:2511.12116) rather than trusting declared
   dates. Llama-3.1-8B is already on the public leaderboard (declared 2023-12, detected 2023-03; §4.1).
   Qwen2.5 has no unambiguous official cutoff declaration, so its primary bound is fixed at the official
   2024-09-19 release date (LCB-post begins 2024-09-20 at day resolution).
   Both final Olmo Instruct model cards state `Date cutoff: Dec. 2024`; month-level conservatism makes
   2025-01-01 the first eligible post-cutoff day. Direct searches of the public pretraining and
   post-training corpora remain necessary for item-level contamination ground truth, not for setting
   this operational time boundary.
5. **Measure residual contamination via TRACER (arXiv:2605.24079), against the Olmo3 pretraining
   corpus** — the only corpus in the design open enough to run it on: TRACER is defined as a function of
   a training corpus and a test set, and Qwen2.5's and Llama-3.1's corpora are closed (§4.5.2). This is
   a prerequisite for Q1b specifically (§4.5.2) and can proceed in parallel with step 6, since Q1a does
   not require it. TRACER has no confirmed public code release; we reimplement it from the paper's own
   specification (its appendix publishes the prompts for all three LLM stages, the embedding model, and
   the triage thresholds), with a retrieval pre-stage (n-gram/BM25 top-k candidates per benchmark item)
   in front, since exhaustive pairwise comparison against a pretraining-scale corpus is infeasible.
   **For the Olmo3 arm, also derive ground-truth labels directly** from the released training data — the
   pretraining corpus *and* the post-training (instruction-tuning) sets, both public for Olmo3, since the
   instruct checkpoints (§4.1) can absorb benchmark items at either stage. We run all three open-data
   detection families in arXiv:2404.00699's taxonomy rather than choosing one, because on this corpus they
   are expected to disagree sharply, and the disagreement is itself the measurement:

   - **(i) Instance-level string matching** — exact and near-exact *n*-gram overlap between each benchmark
     item and the corpus, via a suffix-array/FM-index over the training data (candidate implementation:
     infini-gram; whether a public index exists for Olmo3's corpus release, or whether one must be built,
     is an open engineering item). Decontamination in Olmo 3's own pipeline is stage-specific, not
     corpus-wide (arXiv:2512.13961): the midtraining/long-context stages (~150B tokens) and all
     post-training stages are filtered against exactly HumanEval, MBPP, and LiveCodeBench (the OLMES
     suite used for the midtraining filter, and the post-training evaluation set, both name these three
     benchmarks explicitly), but the bulk pretraining stage (~5.9T tokens, over 97% of the token budget)
     is not — the report concentrates decontamination effort late in training on the stated rationale
     that memorization occurs most strongly there. Run against the **bulk pretraining stage**, family (i)
     is therefore not pre-suppressed and supplies a genuine lower bound, not a spurious *e* ≈ 0. The
     opposite risk — a near-zero result that just re-measures the corpus builders' own filter rather than
     the absence of exposure — applies if family (i) is instead run against the midtraining or
     post-training slices, which *are* filtered against these benchmarks; reporting *e* without noting
     which stage it came from would risk exactly the spuriously optimistic row this step exists to
     prevent, just from the opposite source than originally assumed here.
   - **(ii) Surface- and semantic-level program matching** — the primary source of the labels. We follow
     the pipeline of arXiv:2403.04811, which addresses exactly this problem for exactly these benchmarks:
     edit distance for surface similarity plus AST-based similarity for semantic equivalence, applied with
     a sliding window over the corpus. Because that work evaluates HumanEval and MBPP against
     pretraining-scale corpora and releases its matching outputs, its thresholds and pipeline transfer here
     with minimal adaptation, and its reported rates supply a prior for what to expect.
   - **(iii) Paraphrase detection** — following the retrieval-then-LLM-judge design of arXiv:2311.04850
     (embedding retrieval of top-*k* candidates, then a strong-model judgment on semantic equivalence),
     applied with particular attention to the **post-training sets**, where rephrased benchmark items are
     most likely to appear and where that work reports its own positive findings on instruction data.

   Report *e* from each family separately. The design's operative *e* is the one from the strongest
   method that completes, not the mean, and the spread between (i) and (ii)–(iii) is reported as a
   descriptive result: it quantifies how much contamination survives lexical decontamination. §2.4's
   finding of 78% semantic duplication in this corpus's CodeForces-derived data (arXiv:2602.12413)
   predicts a large spread, and a small one would be the surprising outcome worth reporting.

   Score both the pre/post-cutoff proxy and the TRACER reimplementation's output against these labels.
   This yields the measured *e* of §4.5.2 and, separately, a fidelity measurement for the reimplementation
   itself against known-contaminated items — the only point in the design where either can be checked
   rather than assumed. Note the scope limit recorded in §6: these labels are specific to Olmo3 and
   validate the *method*, not the labels of the closed-corpus arms.
6. **Pilot** (§4.7): Qwen2.5-7B and Olmo3-7B, BNB-nf4 arm first. Precondition: measure the CDD baseline AUC
   gate (§4.6) before interpreting any Q1b pilot numbers; if the gate fails, switch the primary detector to
   probability-based methods for the remainder of the pilot. **Also record measured throughput** (items per
   hour per precision, separately for CDD's multi-sample generation and for the single-pass log-probability
   scoring of §4.4) and convert it into a wall-clock estimate for the full run. One decision depends on this
   number and cannot be made without it: whether CDD's sample count *n* must be reduced — which, per §4.4,
   costs precision only in the CDD arm.
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
   - *Boundary sensitivity (pre-specified, §4.2):* re-run the Q1b/Q2 label assignments under each
     non-corpus-tier arm's bracketing cutoff bounds — declared/primary versus detected/sensitivity, roles
     fixed in advance — plus the descriptive ambiguous-window detector-score comparison, which is never
     fed back into labels.

---

## 6. Threats to Validity

- **Observational, not causal.** Contamination status cannot be randomly assigned to off-the-shelf,
  already-pretrained 7B–32B models; it is an observed covariate, not a treatment. Results are reported as
  associations, following arXiv:2501.18771's causally-identified design as an aspirational reference this
  study cannot replicate at this model scale (§2.3).
- **Declared training-cutoff dates may be wrong, and cutoff evidence quality is heterogeneous across
  arms.** Mitigated via LLMLagBench verification (§5, step 4) and, structurally, via §4.2's evidence-tier
  rule: the primary LCB
  contrast uses the most conservative per-arm bound, trading post-cutoff pool size for validity, and the
  per-arm evidence tier is reported alongside results. Residual uncertainty within each tier remains a
  limitation on the pre/post-cutoff split's precision.
- **"Filtered by date" does not guarantee "uncontaminated."** arXiv:2602.12413 and arXiv:2311.04850
  document that semantic duplication and paraphrase evade time- and n-gram-based filtering. We do not
  claim LiveCodeBench-post is a clean ground truth; we describe it as "lower-contamination" and use
  TRACER (§5, step 5) to measure, rather than assume, residual contamination in every condition, feeding
  that measurement into the label-noise correction in §4.5.2.
- **Extrapolation risk from arXiv:2603.03203, on two independent dimensions.** *Scale:* that paper's
  findings are established at 70M–410M, roughly one to two and a half orders of magnitude below this
  design's 7B–32B range, and the paper itself disclaims extrapolation. *Mechanism:* separately from scale, the
  contamination in that paper is **injected via LoRA fine-tuning**, whereas the contamination this design
  studies **arises naturally during pretraining** (§2.4). This matters because the threshold that governs
  CDD's behavior is stated there in terms of the *absolute number of trainable parameters* — a quantity
  with no clean analogue for pretraining exposure, where there is no adapter, no rank, and no bounded
  training duration to count. We treat CDD's operability at this scale as an open empirical question gated
  by pilot measurement (§4.6), not as an assumption, and report the answer (whichever direction it goes) as
  a standalone contribution. The gate measures the **composite** of both extrapolations and cannot separate
  them: a failed gate does not tell us whether CDD failed because of scale, because the memorization was
  never adapter-shaped to begin with, or both.
- **Olmo3's ground-truth labels are model-local.** The corpus search in §5, step 5 establishes
  contamination status for Olmo3 only. Qwen2.5's and Llama-3.1's training corpora remain closed, so those
  arms keep the proxy labels and their unmeasured error rate. Olmo3's labels therefore validate the
  *labelling method* — how well a release-date split tracks actual corpus membership on this benchmark
  family — and must not be presented as ground truth for the other arms. A measured *e* on the Olmo3 arm
  transfers to the others only under the assumption that the proxy behaves similarly against a different
  corpus, which we state as an assumption rather than a result.
- **Scale mismatch in the bnb-nf4 effect-size prior.** The 32% BNB-nf4 drop cited in §2.7 and §4.3 was
  measured in arXiv:2505.20276 on **Llama-3.1-70B**. Our Llama arm, Llama-3.1-8B, shares the family and
  data recipe but is roughly 9× smaller, and quantization fragility is known to vary with scale. The
  prior therefore identifies the *family* of the expected effect, not its magnitude; the nf4 arm's
  largest-expected-effect status rests on the calibration-free-vs-calibration-based distinction (§4.3),
  not on this per-model figure, and the pilot (§4.7) measures the actual effect before the full run is
  sized.
- **Cutoff uncertainty on the Llama-3.1-8B arm.** LLMLagBench's detected knowledge-drop changepoint
  (2023-03) precedes the declared cutoff (2023-12) by nine months (§4.1, §4.2). We adopt the declared
  date as the conservative exposure boundary and quantify — rather than merely flag — the risk via the
  pre-specified boundary sensitivity analysis of §4.2: the arm's label-dependent analyses are run under
  both bounds and the divergence between runs is itself reported. Note the failure direction is benign:
  if the effective boundary is earlier than declared, the declared-boundary run's suspect condition is
  diluted with clean items, which *attenuates* effects toward null (the same direction as label noise,
  §4.5.2) rather than manufacturing false positives.
- **Base-rate confound between conditions** (§4.5.3) is mitigated on the log-odds scale under a
  constant-odds-ratio assumption, which is itself validated (not assumed) via difficulty stratification.
  If that assumption fails, the design falls back to a difficulty-matched comparison rather than reporting
  an uninterpretable interaction term.
- **Proxy contamination labels carry error** until TRACER measurement is complete (§4.5.2); this
  attenuates, rather than adds noise to, the true effect being measured in Q1b, and is addressed by
  treating TRACER measurement as a Q1b prerequisite rather than an optional check.
- **TRACER is a reimplementation, not the authors' code.** No public code release is confirmed for
  TRACER; §5, step 5 rebuilds it from the paper's published prompts, embedding model, and thresholds.
  Reimplementation fidelity is therefore itself a threat — mitigated, but only on the Olmo3 arm, by
  scoring the reimplementation against the corpus-search ground truth (§5, step 5). Its accuracy on the
  closed-corpus arms' conditions is an assumption inherited by every TRACER-dependent quantity: the
  measured *e*, and Q1b's label-noise correction.
- **Scope limits.** Five models (7B–32.5B), four quantization configurations, code generation only,
  predominantly Python. Findings need not generalize to other architectures (e.g., MoE), other
  tokenizers, or other domains (e.g., natural-language QA), and we do not claim they do.

---

## 8. Limitations

- No causal claims are possible without random assignment of contamination, which cannot be done on
  off-the-shelf pretrained models (§6).
- Q2 may remain underpowered even after all mitigations in §4.5; if so, it is reported as a
  confidence-interval-bounded secondary result, not as a significance claim, and the paper's contribution
  claim does not depend on it clearing significance.
- CDD may be entirely inoperative at 32B scale (the "floor" failure mode, §4.6); if the pilot gate
  fails, this is reported as Contribution 4, not treated as a design failure requiring a redesign.
- Cutoff evidence quality differs across arms (§4.2): Olmo3's boundary rests on official model-card declarations,
  Llama-3.1-8B's on an externally verified declaration, and Qwen2.5's — absent an unambiguous
  declaration — on the release-date upper bound. The pre/post split's precision is therefore
  arm-dependent; boundary evidence is tabulated per arm and the most conservative bound governs the
  pooled contrast.
- The scope is limited to five code-generation-capable dense transformer models (7B–32.5B) and four
  quantization configurations; no model above 32.5B is tested — a hard single-device compute constraint
  (§4.1) — and generalization beyond this scope, upward in scale included, is not claimed.

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
- arXiv:2511.12116 — *LLMLagBench: Identifying Temporal Training Boundaries in Large Language Models*
- arXiv:2504.14655 — *LeetCodeDataset: Temporal Dataset for Robust Evaluation*

**Effect sizes of contamination**
- arXiv:2501.18771 — *Overestimation in LLM Evaluation* (controlled, machine translation)
- arXiv:2403.04811 — *Quantifying Contamination in Evaluating Code Generation Capabilities of Language Models* (ACL 2024)
- arXiv:2506.02791 — *Rethinking the Effects of Data Contamination in Code Intelligence*
- arXiv:2507.19219 — *How Much Do LLMs Cheat? One-Time-Pad Framework*

**Limits of contamination detection**
- arXiv:2311.04850 — *Rethinking Benchmark and Contamination for Language Models with Rephrased Samples*
- arXiv:2602.12413 — *Soft Contamination Means Benchmarks Test Shallow Generalization*
- arXiv:2402.02823 — *Evading Data Contamination Detection is (too) Easy*
- arXiv:2409.09927 — *Limitations, Inconsistencies, and Oracle Challenges* (title appears truncated in this project's source list; confirm against arXiv before submission)
- arXiv:2603.03203 — *No Memorization, No Detection: Output Distribution-Based Contamination Detection in Small Language Models*

**Code-benchmark contamination**
- arXiv:2605.24079 — *TRACER: A Semantic-Aware Framework for Fine-Grained Contamination Detection in Code LLMs*
- arXiv:2512.13961 — *Olmo 3* (Ai2 technical report; cited for its per-stage training-data decontamination methodology, §5 step 5)
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
- arXiv:2410.16454 — *Catastrophic Failure of LLM Unlearning via Quantization* (ICLR 2025)
- arXiv:2605.15138 — *Forgetting That Sticks: Quantization-Permanent Unlearning via Circuit Attribution*
