# Does Quantization Erase the Evidence? Contamination-Detection Signals Under Post-Training Quantization in LLMs on Code Generation

*[Working title — subject to revision after results are in]*

**Status:** Research design draft (pre-execution). Sections marked **[TBD — pending execution]** are
placeholders to be filled in after the plan in §5 is carried out; nothing below reports actual
experimental results.

Authors: [TBD] · Affiliation: [TBD]

---

## Abstract

Reported accuracy drops after post-training quantization can reflect several mechanisms, including
changes in generalization and in the reproduction of memorized content. Studies of utility-constrained
unlearning show that quantization can reverse knowledge suppression; this does not by itself establish
that quantization preferentially erases benchmark answers in otherwise unmodified models. Related work
has examined quantization and membership inference in code LLMs, and verbatim extraction in smaller
language models (§2.8). Separately, controlled contamination experiments on 70M–410M Pythia models find
that CDD (*Contamination Detection via output Distribution*, which scores the peakedness of a model's
output distribution) often fails when fine-tuning does not produce sufficient output concentration, while
probability-based methods retain signal in some of those conditions. These findings motivate a paired
comparison of CDD, perplexity, and Min-k% Prob (the mean log-probability of an item's least likely
tokens) under quantization; their behavior at 7B–32.5B is not assumed.

We present a pre-execution observational design. Five instruction-tuned checkpoints spanning 7B–32.5B are
scored on LiveCodeBench, HumanEval and MBPP+ at bf16 and three post-training quantization settings
(BNB int8, BNB-nf4, AWQ-int4). Each LiveCodeBench item carries a temporal exposure proxy: its publication
date is compared with that model's declared training boundary to label it `possible-exposure` or
`shared-clean-control`, which marks possible rather than verified exposure. The primary question (Q1)
measures the mean within-item shift of each detector's score (Q1a) and changes in how well those scores
separate the two proxy groups, including a pre-specified reversal of the detector-family ranking (Q1b).
Its power depends on the actual effect and covariance structure; the planning tables do not guarantee
power for the confirmatory detector-family ranking-reversal test. The secondary question (Q2) estimates a
quantization × exposure-proxy interaction in pass@1 on the conditional log-odds scale, with explicit
coding and difficulty diagnostics. The four-test confirmatory family is restricted to one model
(Qwen2.5-32B-Instruct) and one contrast (bf16→BNB-nf4); all other models, precisions and benchmarks are
exploratory. We report the planning calculations, that confirmatory family, a reusable item-level dataset
of detector and task scores, and a separation between engineering validation and study data. Neither
temporal-proxy AUC nor Q2 identifies the causal contribution of memorization to accuracy loss.
**Results: [TBD — pending execution of §5].**

---

## 1. Introduction

Post-training quantization (PTQ) is now standard practice for deploying large language models, and a
substantial literature reports its effect on downstream accuracy. That literature generally treats accuracy
drops as measurements of degraded *capability*. This framing carries an assumption that such reports rarely
test directly: that benchmark performance under full precision reflects capability rather than recall of
memorized training data.

Two independent lines of evidence suggest this assumption should not be taken for granted.

**Quantization's rounding grid can erase small parameter differences.** Studies of machine unlearning — the
practice of suppressing specific knowledge in a trained model without full retraining — find that "erased"
knowledge often resurfaces after quantization: one study reports that, averaged over unlearning methods
that carry utility constraints, retained forgotten knowledge rises from 21% at full precision to 83%
after 4-bit round-to-nearest (RTN) quantization, measured on the MUSE benchmark
(arXiv:2410.16454, §4.2; the qualifier is the source's own — it singles out unconstrained gradient ascent
as the one method whose apparent forgetting after quantization is misleading, because there the
forgetting comes from a complete loss of model utility). A mechanistic follow-up explains why:
the per-parameter updates that gradient-based unlearning produces are 47–828× smaller than a single
NF4 (4-bit NormalFloat) quantization bin, so quantization's rounding simply erases them (arXiv:2605.15138,
"Forgetting That Sticks"). That range is not a frequency of occurrence but the ratio of per-parameter
update RMS to the NF4 bin width (8.4×10⁻⁴) for two gradient-ascent baselines on
Llama-3.1-8B-Instruct / WMDP-bio — Global GA at about 1/828 and Surgical GA at about 1/47 (Appendix M,
Table 15). In those studies quantization therefore *restores* the original memorization
rather than removing it; what carries over to our setting is the size argument, not that direction. The
connecting hypothesis is that weakly exposed benchmark content may likewise be held in weight differences
smaller than a bin, in which case rounding would remove it as well. The same argument makes the opposite
prediction for strongly memorized content, whose parameter differences exceed the bin width and survive
rounding. Because both outcomes are plausible from the same mechanism, the confirmatory tests in §4.5.6 are
two-sided. Neither study is direct evidence that naturally memorized benchmark answers are preferentially
erased; we use the mechanism as motivation for testing signal stability, and discuss direct extraction
evidence separately in §2.8.

**Contamination detectors are not interchangeable.** Sela's replication on 70M–410M Pythia models
finds that CDD frequently performs at chance under contamination injected by fine-tuning, while
probability-based detectors outperform it in conditions with detectable signal (arXiv:2603.03203).
The author explicitly cautions that the findings "should not be extrapolated to larger scales without
further investigation." The positive 7B evidence cited there belongs to Dong et al.'s original CDD
study, not to the replication. Neither study establishes CDD's behavior on our naturally exposed
7B–32.5B Instruct checkpoints. The reported capacity threshold involves model size, trainable
parameters, and training duration in the tested fine-tuning regimes (§2.4).

Contamination assessments made on full-precision checkpoints may not transfer to quantized deployments
if detector scores shift with precision. Quantization is a common deployment transformation, and its grid
structure can disturb memorized traces, so whether a full-precision contamination verdict survives that
transformation is itself an open measurement question.

Prior work has already crossed quantization with training-data inference — membership inference in code
LLMs, and verbatim extraction in smaller language models — but has not compared output-distribution
peakedness against the probability family on the same items under quantization at 7B–32.5B; that is the
gap this design addresses (§2.8).

This motivates our specific question: **if quantization
measurably disturbs memorized traces, and contamination detectors differ in how much they depend on those
traces, does quantization differentially reshape what different contamination detectors see?**

A related but harder-to-power question — whether the
*quantization-induced conditional log-odds drop* differs between possible-exposure and shared-control proxy conditions —
is under-powered at benchmark-imposed sample sizes and is therefore **secondary** (§3.0, §4.5).

**Contributions (stated as design commitments, to be confirmed or refuted by execution):**

1. A paired measurement, at 7B–32.5B scale, of how much post-training quantization shifts each
   contamination-detection signal's own score — output-distribution peakedness (CDD) and the probability
   family (perplexity, Min-k% Prob) — on the same items at each precision. This measures each detector
   separately and does not by itself compare detectors with one another (Q1a).
2. A pre-specified test of whether quantization reverses the *ranking* between the two detector families
   in temporal-proxy separation. This is the design's only confirmatory comparison *between* detector
   families; changes in the gap without a reversal are exploratory (Q1b).
3. A bounded, log-odds-scale estimate of the quantization × exposure-proxy interaction on pass@1 in code
   generation, reported as an association rather than a causal effect, with explicit scale assumptions and difficulty
   diagnostics (Q2).
4. A reusable item-level dataset — pass@1, partial credit, token log-probability, and three detector
   scores, crossed with quantization technique, precision, and model. Its only exposure annotations are
   the model–item temporal proxy labels of §4.2 and, for the two Olmo3 arms, the corpus-reference status
   of §5, step 5; it carries no verified contamination labels, so it supports comparing methods on common
   items rather than serving as contamination-detection ground truth.
5. Evidence on how much CDD's score shifts under quantization and on temporal-proxy separation at 32B scale; proxy AUC alone does not establish operability against verified contamination.

Contributions 1 and 2 are the primary claim; contribution 3 is secondary. The confirmatory evidence for
contributions 1 and 2 comes from one model and one contrast — Qwen2.5-32B-Instruct, bf16→BNB-nf4 — and
everything else in the design is exploratory (§3.2, §4.5.6).

---

## 2. Related Work

### 2.1 Contamination: definitions, causes, mitigation (surveys)
Data contamination — benchmark items appearing, verbatim or paraphrased, in a model's training data — is
a mature research area with several existing surveys. arXiv:2404.00699 (*A Comprehensive Survey of
Contamination Detection Methods in Large Language Models*, TMLR 2025) has the broadest detection-method
coverage; arXiv:2502.14425 is the more recent general survey of definitions, causes, and mitigation but
covers detection methods less exhaustively; arXiv:2605.26133 is a unified survey of data contamination
and membership inference under a single "pretraining data exposure" framing; arXiv:2502.17521 motivates the shift from static to dynamic
benchmarking, which we invoke to justify our use of LiveCodeBench.

### 2.2 Temporal-split contamination measurement
Roberts et al. (arXiv:2310.10628, *Data Contamination Through the Lens of Time*) examine
benchmark performance over release dates and describe their use of GPT training cutoffs as a
natural experiment. This provides methodological precedent for using publication time, not automatic
causal identification for our model-specific LiveCodeBench split (§4.2). Our temporal labels remain
observational proxies, subject to date-related difficulty and topic differences. LiveCodeBench
(arXiv:2403.07974) provides dated problem collection suited to this comparison.
arXiv:2504.14655 (*LeetCodeDataset*) applies the same temporal-split principle to LeetCode problems, at a
published split date of July 2024. We use LiveCodeBench because its continuously updated dated releases
extend past this design's common 2025-01-01 boundary (§4.2); we compared no matched pools and therefore
make no claim about which collection is larger. The existence of an independent second instance
corroborates the design pattern.
Because publicly declared training-cutoff dates can be wrong or absent, we additionally rely on
arXiv:2511.12116 (*LLMLagBench*), which estimates probable temporal knowledge boundaries from
responses about recent events. We use it as an independent behavioral diagnostic (§5, step 4), not
as verification of the last training-data date or of non-exposure after a detected changepoint.

### 2.3 Effect sizes of contamination
arXiv:2501.18771 provides a controlled, causally-identified estimate of contamination's effect by directly
pretraining 1B/8B models on machine-translation data with contamination injected at controlled stages,
scales, and formats. We treat this as the methodological reference point for causal identification, but
**cannot replicate its design**: it requires pretraining from scratch, and our study uses off-the-shelf
7B–32.5B models in the code domain. We can therefore only approximate its causal design observationally
(§6). arXiv:2403.04811 quantifies contamination's effect size in code generation. arXiv:2506.02791
constructs fine-grained contamination settings across three code tasks — translation, generation and
summarization — and notes that most prior work measures only sample-level contamination and
under-counts the more common partial-contamination case; its headline result is that most of those
settings do *not* produce significant overestimation, the exception being paired contamination in models
that are pre-trained and then used for inference directly. arXiv:2403.04811 is additionally a
*methodological* source for this design, not only an effect-size one: its surface-plus-AST matching
pipeline, developed for HumanEval and MBPP against pretraining-scale corpora, is what we adopt for the
Olmo3 corpus-reference positive-evidence search in §5, step 5. arXiv:2507.19219 offers a one-time-pad-based
framework for quantifying benchmark-score overestimation generally.

### 2.4 Limits of contamination detection
This subsection is the paper's most load-bearing prior work. arXiv:2311.04850 shows n-gram-based
decontamination filtering is trivially evaded by paraphrase or translation. arXiv:2602.12413 (*Soft
Contamination Means Benchmarks Test Shallow Generalization*) extends this: semantic (non-lexical)
duplication is undetectable by n-gram matching and was found pervasively in the Olmo3 pretraining corpus,
including CodeForces (77.5% of benchmark problems had at least one semantic duplicate among their top-100 retrieved training-data candidates) — this is the strongest available
objection to treating any time-filtered benchmark as "clean," and we address it directly in §6. Because
Olmo3 is one of the models in our own design (§4.1), this result is not borrowed evidence about some other
model's corpus but a direct prior on possible exposure in one of our arms. The same corpus openness provides
confirmed-positive evidence for that arm, while incomplete retrieval and unobservable training records
prevent a complete binary exposure label (§4.5.2).
arXiv:2402.02823 and arXiv:2409.09927 further document that intentional contamination is easy to hide
from detectors, and that existing detectors disagree with each other on modern LLMs.

**arXiv:2603.03203** (*No Memorization, No Detection*) is the anchor citation for our primary research
question, and because its claims are easy to over-generalize, it merits a precise account of what it does
and does not establish.

- **What it shows:** Sela (Tel Aviv University) replicates the CDD contamination detector of Dong et al.
  (2024) on 70M–410M models, injecting controlled contamination on GSM8K/HumanEval/MATH by fine-tuning
  across three regimes — LoRA r=8, LoRA r=256, and full fine-tuning — at 3 and 20 epochs. Across most of
  the conditions tested, CDD collapses to chance-level accuracy, even when the underlying data is
  "detectable by simpler methods." The paper reports that probability-based detectors (perplexity, Min-k%
  Prob) outperform CDD in every condition where *any* method exceeds chance. The paper's strongest
  supporting quote for using probability-based methods
  as the primary detector family is: *"The gap is largest precisely where it matters most: at low
  contamination levels and under parameter-efficient fine-tuning, where CDD is uniformly at chance but
  probability-based methods already show signal."* We deliberately do not paraphrase this as
  "probability-based methods work wherever CDD fails." The paper's abstract states the comparison
  conditionally — "outperform CDD in all conditions where any method exceeds chance" does not imply
  probability-based methods always exceed chance. Its supporting evidence has a stated scope: Table 2
  covers 27 conditions on Pythia-410M at 3 epochs (three fine-tuning regimes × contamination levels
  c ∈ {1, 5, 10} × GSM8K, HumanEval and MATH), with "above chance" defined there as accuracy above 0.55,
  and CDD clears that bar in 7 of the 27 conditions against 26 for perplexity and 25 for Min-k% Prob.
  Read literally the "*any* method" clause also has to accommodate the N-gram baseline, which has training-corpus
  access: in the HumanEval, LoRA r=8, c=1 cell of that table, N-gram scores 1.0 while CDD, perplexity and
  Min-k% Prob all score 0.53, so probability-based methods do not outperform CDD in that condition. While
  the paper's Conclusion goes further
  (*"…including those where CDD fails entirely"*), that clause establishes only that the set of such
  conditions contains cases of outright CDD failure, not that it exhausts them. The quoted sentence above
  supports the design decision we actually need — probability-based detectors as primary, CDD as a
  comparison arm — without the stronger claim.
- **What governs CDD's failure:** CDD succeeds only when fine-tuning produces verbatim memorization. The
  paper's stated finding is that CDD's "effectiveness depends critically on whether fine-tuning produces
  verbatim memorization," because CDD "requires output distribution collapse to succeed." Whether that
  collapse happens is itself thresholded: "CDD accuracy transitions sharply from chance to >90% as
  fine-tuning capacity crosses a threshold" that "depends on the interaction of model size, adapter rank,
  and training duration." In that paper's runs, the regime that produces collapse is full fine-tuning
  (CDD accuracy 0.955 at 3 epochs on GSM8K, Pythia-410M, contamination level 10), not LoRA r=8 at the same
  duration, where CDD is at chance. The paper's Discussion states that *"the relevant factor is not the LoRA
  rank itself but the absolute number of trainable parameters,"* but rank and parameter count are not the
  whole of its threshold: the threshold is stated as an interaction of model size, adapter rank and
  training duration, and longer training partly substitutes for low rank — LoRA r=8 at 20 epochs reaches
  0.920 on GSM8K at contamination level 10, comparable to LoRA r=256 at 3 epochs (its §4.3). The absolute-capacity comparison the
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
  threshold is simply unknown**. Moreover, a temporal-proxy AUC near 0.5 cannot establish where a
  model lies relative to a true-memorization threshold. Even if CDD's AUC remains 0.5 at both
  precisions, probability-based AUCs can change and a relative ranking can reverse. Only a zero
  value of the particular contrast under test removes its nonzero effect; CDD's chance-level AUC
  alone does not make all of Q1b undetectable. All detectors remain in the frozen analysis (§4.7).

### 2.5 Code-benchmark-specific contamination
arXiv:2605.24079 (*TRACER*) models code contamination as three levels of semantic overlap
(functionally identical / nearly identical / shared logic). Only the first two are forms of duplication:
shared logic covers task pairs that address different objectives while relying on the same core
algorithmic or reasoning strategy. It is the tool we use to seek semantic
positive-match evidence beyond the pre/post-cutoff date proxy
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
The effect-size reference for this design's power analysis (arXiv:2505.20276) reports that, on average,
8-bit quantization preserves accuracy (≈0.8% drop) while 4-bit methods lead to substantial losses (drops
of up to 59%) — including a 32% drop for Llama-3.1-70B under calibration-free BNB-nf4 on the same task on
which Qwen-2.5-72B remains robust. This result is for **long-context (>64K token) evaluation**, not code
generation, and cannot be transferred directly; we use it only as the effect-size reference for our
BNB-nf4 arm (§4.3), where it is our best available prior for "worst case." That study also evaluates
models at the sizes this design uses — Llama-3.1-8B and Qwen-2.5-7B and 32B — and includes AWQ-int4 among
its five quantization methods, so matched-size and matched-method results exist within the same source.
The 32% figure is the only per-model number its abstract states; we quote no corresponding figure at
8B/7B/32B, so the number carried into §4.3 remains a 70B one. For code generation specifically, three
studies point the same way while measuring different quantities. arXiv:2503.07103 replicates earlier
quantization work on code LLMs up to 34B and reports 4-bit as the precision at which performance is
retained, comparing calibration datasets — including code-specific ones — rather than naming a
quantization method in its abstract. arXiv:2507.09665 applies **AWQ** to CodeLlama and DeepSeekCoder and
reports that functional correctness is preserved along with maintainability and structural simplicity.
arXiv:2506.22776 is a robustness rather than an accuracy study: across four model families from 350M to
33B it perturbs input prompts adversarially and adds noise to model weights, and its 51.59% versus 42.86%
are the **proportions of its adversarial experiments** in which the quantized and the full-precision model
respectively showed the better resilience — not accuracies. Together these narrow our expected effect
size for the AWQ comparison and — because they are themselves derived largely from
benchmarks whose contamination status is unexamined — are part of this paper's motivation rather than a
reason to expect a large effect (§4.3, §4.5.3).

### 2.8 The gap this paper addresses
Quantization and training-data inference already intersect. Haque et al. (arXiv:2508.00128,
*How Quantization Impacts Privacy Risk on LLMs for Code?*) study quantization, task performance, and
membership inference across Pythia, CodeGen, and GPT-Neo; they report that quantization significantly
reduces membership-inference privacy risk relative to the original model, and that task performance and
privacy risk are positively correlated. Sasi (arXiv:2607.25451, *Bits and Memories: Measuring Verbatim
Extraction Across LLM Quantization*) studies strict extraction of known memorized sequences in
Pythia-160M/410M/1B alongside perplexity, and reports that verbatim memorization falls off faster than
capability at every precision and model size tested, yet that at the largest model studied four-bit
quantization still reproduces most of the memorized sequences while giving up only a few percent of
capability, with the surviving fraction growing with model size. Neither is a paired comparison of
output-distribution peakedness (CDD) against the probability family on the same items under temporal
exposure-proxy labels at 7B–32.5B — the measurement this design adds — and their findings are not assumed
to transfer to our larger Instruct models or temporal proxies.

Our specific contribution is the paired comparison of output-distribution peakedness and fixed-prompt
probability scores, including a pre-specified family-ranking reversal test, across the frozen PTQ
configurations on code benchmarks at 7B–32.5B. We distinguish score movement, temporal-proxy AUC,
verbatim extraction, and verified training membership. We make no exhaustive "first" or
"no published work" claim about the broader quantization–memorization intersection.

---

## 3. Research Questions

### 3.0 Two questions, and why one is primary

**Q1 (primary). How does post-training quantization affect contamination-detection signals?**
Specifically: does quantization differentially modulate peakedness-based detection (CDD) versus
probability-based detection (perplexity, Min-k% Prob), to the point of changing which detector family
has the higher temporal-proxy AUC at a given precision?

- **Q1a.** Does quantization shift detector scores — specifically, is a detector's **mean within-item
  shift** between two precisions nonzero? (Paired comparison, same item scored at each precision.)
- **Q1b.** Does quantization change a detector's ability to separate model–item
  `possible-exposure` from `shared-clean-control` observations (proxy-label AUC), and does it change
  the *ranking* between detector families? This estimand is not presented as AUC against verified
  contamination ground truth.

**Q2 (secondary). Is there a quantization × exposure-proxy interaction in pass@1?**
Does the *conditional log-odds* drop from quantization differ between a possible-exposure proxy condition
(model-specific LiveCodeBench `possible-exposure`) and the shared post-boundary LiveCodeBench control?
The contrast is defined on the conditional log-odds scale, not on raw percentage points (§3.1).
HumanEval and MBPP+ supply separate exploratory suspect-proxy contrasts; they are not pooled.

Q2 is secondary because the primary shared-control
cell has a fixed ceiling of 182 items. Under the conservative unpaired p=0.5 calculation in §4.5.3, even an
infinite suspect cell leaves a **14.7 percentage-point** minimum detectable interaction; using the largest
possible 873-item suspect envelope raises it to **16.1 percentage points**, and every model's actual suspect
subset is no larger (Olmo attains 873). The secondary HumanEval contrast reaches 15.5 points at best, at its
fixed 164-item ceiling. These limits do not adequately resolve illustrative 5–10pp targets. Quantization-in-code studies (§2.7) concern different settings and do not establish our effect size. Q1, in contrast,
has more favorable illustrative planning requirements: Q1a's normal-approximation requirements are
about 87–196 items for paired standardized effects of 0.3–0.2 (§4.5.1; ≈124–279 items at the confirmatory
family's Holm-adjusted α/4, §4.5.6). The two-AUC planning example
at 164 items per label group gives a 0.051 detection limit under its stated assumptions (§4.5.2),
not guaranteed power for C4. Confirmatory tests use the primary model's LiveCodeBench (LCB) items at the
bf16→BNB-nf4 contrast only; every other model and precision is exploratory (§4.5.6).
**Q1 (Q1a and Q1b) is primary; Q2 is secondary.**

### 3.1 Q2 design: a 2×2 comparison

|  | Full precision | Quantized | Difference |
|---|---|---|---|
| **Model-specific possible-exposure proxy** (LiveCodeBench before the arm boundary) | A | B | A − B |
| **Shared post-boundary control** (LiveCodeBench on/after 2025-01-01) | C | D | C − D |

A, B, C, and D in the table denote probabilities. For interpretation of *drops*, define the
conditional log-odds contrast J = [logit(A) − logit(B)] − [logit(C) − logit(D)], comparing cells at
the same random-effect values, not logits of marginal pooled accuracies. Fit each quantized level
against bf16 with Q=0 for bf16, Q=1 for that quantized level, E=0 for shared control, and E=1 for
possible exposure. In logit Pr(correct)=…+β_Q Q+β_E E+β_QE QE, the fitted interaction is **β_QE=−J**.
Report both the coefficient and the drop-oriented contrast with explicit signs (§4.5.5).
The interpretations below refer to J, not directly to β_QE.

- **Near zero:** the estimated conditional log-odds interaction is close to zero. This alone does not
  establish absence of exposure-proxy modulation or equivalence; uncertainty must be reported.
- **Positive:** the conditional log-odds drop is larger in the possible-exposure proxy condition.
  This is compatible with, but does not identify, loss of memorized content.
- **Negative:** the conditional log-odds drop is smaller in the possible-exposure proxy condition.
  This does not by itself demonstrate either an absolute accuracy increase or resurfacing of knowledge.

The sign of J need not match the sign of a raw percentage-point interaction. For example, at fixed
random effects, baseline probabilities A=0.85 and C=0.35 and respective log-odds drops 0.30 and 0.20
produce J=+0.10 but a raw interaction of approximately −0.165pp. Consequently a positive J does not
mean the raw accuracy drop is larger. A direct B-vs-A comparison is descriptive; even an absolute
increase with verified exposure does not identify the utility-constrained unlearning reversal mechanism
in arXiv:2410.16454 without evidence of prior suppression and its reversal.

### 3.1.1 Interpretation of Q2

A common conditional log-odds effect can produce a nonzero raw percentage-point interaction.
Its sign depends on baseline probabilities and the difficulty distribution: the 0.85/0.35 worked
example in §4.5.3 is negative, not a universal sign rule. Consequently neither sign is inherently
protected from scale artifacts or exposure–difficulty confounding. Interpret J only with the
pre-specified scale and difficulty diagnostics. Near-zero estimates with wide intervals are
inconclusive; this protocol specifies no equivalence margin and therefore makes no equivalence claim.

Q1 does not depend on Q2's sign or significance.

### 3.2 Scope of claims

The framing that opens §1 — that a reported quantization accuracy drop is read as degraded capability,
while full-precision benchmark performance may instead reflect recall of memorized training data — is the
paper's *motivation*, not a result. Neither Q2's sign nor the direct B-vs-A comparison
identifies loss of memorization; both can reflect other changes in model behavior or item composition.
The paper's *primary contribution* claim, stated in the abstract and §1, is Q1; the §1 list also includes
the secondary Q2 estimate and the released item-level dataset, which do not depend on Q1's outcome. We keep
motivation and contribution separate throughout to avoid the failure mode of claiming to have measured
something the design cannot actually power.

**Confirmatory scope.** The four confirmatory tests cover one model and one contrast:
Qwen2.5-32B-Instruct, bf16→BNB-nf4 (§4.5.6). Results for the other four models, for BNB int8 and
AWQ-int4, and for HumanEval and MBPP+ are exploratory and are reported with intervals rather than
confirmatory significance claims.

**What rejecting C1–C3 does and does not mean.** C1–C3 test the mean within-item nf4−bf16 shift of a
single detector's score over all 1,055 LCB items, with no exposure label used. Rejecting one of them means
that the detector's score scale — and therefore any decision threshold calibrated at bf16 — moves with
precision. That movement applies to every item alike and is not separated from exposure-related movement,
so it is not by itself evidence that a contamination signal has weakened. Whether the exposure-related
separation survives quantization is addressed only by Q1b, which uses the proxy labels.

**What this design can and cannot answer.** It can report how far each detector's score moves between
precisions, how well those scores separate the temporal proxy groups at each precision, and whether the
ranking between detector families reverses. It cannot demonstrate that a *verified* contamination signal
is erased: no arm has a verified exposure label, the Olmo3 corpus search yields positive matches only, and
a searched non-match is not a verified negative (§4.2, §4.5.2). It equally cannot demonstrate that no
erasure occurs, because Q1 declares no equivalence margin; a nonsignificant C1–C4 result is inconclusive
rather than evidence of stability.

---

## 4. Method

### 4.1 Models

| Model | Size | Baseline precision | Role | Notes |
|---|---|---|---|---|
| Qwen2.5-32B-Instruct | 32.5B | bf16 | Primary | Dense, GQA+RoPE; no official QAT checkpoint exists, so naive-PTQ comparisons are uncontaminated by a QAT confound |
| Qwen2.5-7B-Instruct | 7B | bf16 | 7B size axis | Lower execution cost than the 32B arm and a secondary "does the effect scale with model size" probe; scored rows enter the study only in the frozen main run |
| Llama-3.1-8B-Instruct | 8B | bf16 | Size axis + external knowledge-boundary diagnostic | The only main-analysis model with an independent LLMLagBench knowledge-boundary estimate (arXiv:2511.12116): declared 2023-12, detected knowledge-drop changepoint 2023-03. We use the declared (later) date as the conservative contamination boundary; the detected/declared gap and its consequence for this arm's LCB pre-cutoff pool are discussed in §4.2. Family tie to arXiv:2505.20276's BNB-nf4 fragility prior. That study also evaluates Llama-3.1-8B, but the 32% drop is the only per-model figure its abstract states and that figure is a Llama-3.1-**70B** result (§2.7), so the number carried into §4.3 comes from a model roughly 9× larger than this arm — a weak prior rather than a confirmed expectation (§6) |
| Olmo3-7B-Instruct | 7B | bf16 | Corpus-reference positive evidence + size axis | Fully open training data across all stages (pretraining and post-training), enabling direct positive-match evidence without treating searched non-matches as verified negatives. Dense transformer; architectural and training differences from the other families remain |
| Olmo3.1-32B-Instruct | 32B | bf16 | Corpus-reference positive evidence + size axis | Official 32B final Instruct release (`allenai/Olmo-3.1-32B-Instruct`), with a nominal weight-memory footprint similar to Qwen2.5-32B-Instruct |

All five arms use the instruction-tuned (\*-Instruct) releases: code-generation pass@1 under
instruction prompts is the measured quantity, the illustrative base rates in §4.5.3 are
instruction-tuned figures, and the LLMLagBench diagnostic (§5, step 4) probes instruct checkpoints.
Shorthand names elsewhere in this document (e.g., "Qwen2.5-7B") refer to these Instruct checkpoints.
"Olmo3" written without a size refers to the two open-data arms jointly — Olmo3-7B-Instruct and
Olmo3.1-32B-Instruct — and never to a single checkpoint. The two are separate releases whose bulk
pretraining mixes are different datasets, so every corpus-reference search in §5, step 5 is run and
reported per checkpoint rather than once for "the Olmo3 corpus": a confirmed match found in one arm's
training data is evidence about that arm only.
An instruct checkpoint also widens the contamination surface — benchmark items can enter through
post-training (instruction-tuning) data as well as pretraining — which is why the Olmo3 corpus-reference
search in §5, step 5 covers both stages.

QAT-shipped models (e.g., Gemma-family official QAT checkpoints) are excluded entirely: an official QAT
checkpoint degrades structurally differently from naive PTQ, and the available QAT models add further
uncontrolled confounds (thinking-mode toggles, multimodality). A QAT-vs-PTQ comparison on such a model
is deferred to future work; the format of the shipped checkpoints (llama.cpp q4_0) would also require a
second inference stack, whose numerics differences would confound the very contrast of interest.

Architecture is not a controlled comparison axis. All arms are dense transformers, but this does
not equate their attention, normalization, tokenizer, pretraining, or post-training configurations.
Size is the intended descriptive model-comparison axis, not an isolated causal factor. Cross-family
size trends remain confounded by these differences. Olmo corpus transparency is an operational
selection property enabling positive-match evidence, not an effect-modifying comparison axis.
Within-model PTQ comparisons use the same frozen checkpoint and bf16 baseline.

**Compute footprint.** The available hardware is a single H100 (80 GB), with a single H200 (141 GB)
obtainable on request. Weight footprints, before KV cache and activations:

| Model | bf16 | int8 | int4 (nf4) | Fits a single device at bf16? |
|---|---|---|---|---|
| Qwen2.5-7B / Olmo3-7B / Llama-3.1-8B | ~14–16 GB | ~7–8 GB | ~4–5 GB | Yes (H100) |
| Qwen2.5-32B / Olmo3.1-32B | ~64–65 GB | ~32 GB | ~18 GB | Yes, tightly (H100, ~15 GB left for KV cache); comfortably on the H200 |

Every figure in this table is an estimate derived from parameter counts, not a measurement. The int8
and int4 columns cover the **quantized linear layers only**: both bitsandbytes rungs leave the
embedding matrix and the language-model head in bf16 (§4.3), and those two matrices are large at these
vocabulary sizes, so each arm's actual quantized footprint is larger than the column shows. The margins
below are read accordingly — the ordering of the rungs and the single-device conclusion do not depend on
the exact figures, and actual memory use is among the quantities engineering validation checks (§4.6).

Every arm therefore runs its complete quantization ladder — bf16 baseline included — on a single
available device. No arm requires a baseline at a different precision from any other, which keeps
Q1a's bf16-anchored within-model contrast directly comparable across all five models. Models above
32.5B are excluded from the design as a hard compute constraint (a single device cannot hold a 70B-class
bf16 baseline), and this scale ceiling is recorded as a scope limitation in §7.

### 4.2 Data: contamination axis

| Axis | Condition | Available *n* | Rationale | Note |
|---|---|---|---|---|
| Primary suspect proxy | LiveCodeBench, model-specific `possible-exposure` | Qwen2.5: 690; Llama-3.1 primary: 326; Olmo3: 873 | Same source and format as the shared control; publication time defines possible exposure, not confirmed contamination | `release_v6` availability counts; Llama sensitivity has 0 possible-exposure items and is not estimable |
| Primary shared control | LiveCodeBench, `shared-clean-control` | 182 available | On or after 2025-01-01, the first day after the latest model-level cutoff | Clean only in the temporal sense defined below; Q2 is secondary and interval-focused |
| Secondary suspect proxy | HumanEval | 164 (hard ceiling) | Released 2021; plausible exposure for all five models, not confirmed membership | Exploratory Q1a only; no within-benchmark control group, so no Q1b here, and not a Q2 cell (§4.5.3) |
| Secondary suspect proxy | MBPP+ | 378 | Separate arm | Exploratory Q1a only; no within-benchmark control group, so no Q1b here. **Not pooled** with HumanEval — different difficulty distributions would reintroduce the base-rate confound *inside* a nominally single condition. The combined n=542 is a sample-size reference only, never a pooled analysis cell. |

The temporal split is the primary proxy axis because it holds source and format fixed and reduces — but
does not eliminate — the difficulty confound in §4.5.3. A publication date before a model's bound means
only that the item **could have been exposed** during training; it does not establish that the item was
actually present. Conversely, `clean-by-model-cutoff` means only that the item falls after the stated temporal
bound, not that semantic duplication or later-stage exposure has been ruled out. We therefore reserve
`contaminated` for corpus-confirmed membership and use `possible-exposure` / `clean-by-model-cutoff` for the
date proxy.

Under LiveCodeBench `release_v6`, the common 2025-01-01 boundary yields an 873-item pre-common candidate
envelope and 182 shared-control items (1,055 total). **The 873 items are not one contamination-suspect
analysis cell shared by all models.** Each arm's suspect cell is the subset published before that arm's
own primary bound; items after an arm's bound but before 2025-01-01 are temporally clean for that arm but
are excluded from the primary shared-control contrast. Both sides of the contrast are therefore small: the
common control has 182 items, and every arm-specific suspect cell has at most 873. Q2 is accordingly
secondary and interval-focused.

**Pre-specified boundary rule.** Cutoff evidence quality differs by arm, so the boundary is defined to
remain valid under the weakest evidence rather than under the most optimistic reading:

| Arm | Cutoff evidence | Evidence tier | Primary first post-boundary date | Sensitivity first post-boundary date |
|---|---|---|---|---|
| Olmo3-7B / Olmo3.1-32B | Official model cards for both final Instruct checkpoints state `Date cutoff: Dec. 2024`, without stating which training stages the date covers ([7B](https://huggingface.co/allenai/Olmo-3-7B-Instruct), [32B](https://huggingface.co/allenai/Olmo-3.1-32B-Instruct)) | Official model-level declaration, stage scope unstated | 2025-01-01 | — |
| Llama-3.1-8B | Declared 2023-12; LLMLagBench detects a knowledge drop in 2023-03 | Declaration + independent behavioral diagnostic | 2024-01-01 (declared cutoff through Dec. 2023) | 2023-04-01 (detected boundary through Mar. 2023) |
| Qwen2.5-7B / Qwen2.5-32B | No unambiguous official cutoff declaration | Release-date bound (weakest) | 2024-09-20 ([official release announcement](https://qwenlm.github.io/blog/qwen2.5/); the 2024-09-19 release day is not a post-boundary date, so an item published that day is `possible-exposure` rather than dropped) | — |

For an immutable released checkpoint, its release date bounds when its training could have occurred.
This does not rule out private prepublication versions, earlier equivalents, or later calibration-data
exposure; the bound is applied only to the recorded public release of each benchmark item. Let
*t*<sub>*i*</sub> be an item's publication date and *c*<sub>*m*</sub> the first post-boundary date for
model *m*. For LiveCodeBench, *t*<sub>*i*</sub> is the release's own `contest_date` field, read as the
calendar date it records: `release_v6` ships it as a timezone-naive ISO timestamp at midnight, and we
apply no timezone conversion, so both sides of every comparison below are plain calendar dates. The
stored labels are **model–item labels**, not one global item label:

- `possible-exposure` if *t*<sub>*i*</sub> < *c*<sub>*m*</sub>;
- `clean-by-model-cutoff` if *t*<sub>*i*</sub> ≥ *c*<sub>*m*</sub>;
- `shared-clean-control` if *t*<sub>*i*</sub> ≥ 2025-01-01, the latest primary bound across arms; and
- `boundary_ambiguous = true` when an item lies between an arm's sensitivity and primary bounds.

As written these three conditions are not mutually exclusive: `shared-clean-control` is a **subset** of
`clean-by-model-cutoff`, since an item at or after 2025-01-01 is also at or after every arm's own bound.
Where the three are tabulated as a partition of the 1,055 LCB items, each item is counted once under the
most specific label that applies — `shared-clean-control` first, then `clean-by-model-cutoff`, then
`possible-exposure` — so a tabulated `clean-by-model-cutoff` count is the post-bound remainder with the
shared control already taken out, not the full set of items satisfying *t*<sub>*i*</sub> ≥
*c*<sub>*m*</sub>.

The interval after an arm's own bound but before 2025-01-01 is retained as
`clean-by-model-cutoff` metadata but excluded from the primary shared-control contrast. Thus an item can
be `possible-exposure` for Olmo3 yet `clean-by-model-cutoff` for Qwen2.5 or Llama-3.1. The primary pooled
LCB contrast uses each arm's own `possible-exposure` items against the same 182-item
`shared-clean-control`; per-model results are reported before any pooled estimate.

Corpus evidence is stored on a **separate axis** that is recorded alongside the temporal label rather
than merged into it: a model–item pair carries both values, and neither overrides the other. The axis has
three values, recorded per arm and separately for each of the three search families of §5, step 5:

- `confirmed-match` — that family found, in that arm's own released training data, at least one record
  accepted as containing the benchmark item. Family (i) records a match on exact or near-exact *n*-gram
  overlap with the item text; families (ii) and (iii) return candidates, and a candidate becomes a
  `confirmed-match` only once adjudication accepts it as the same problem (§5, step 5).
- `no-match-found` — the family searched the available corpus and accepted no match. This is not renamed
  `clean`: the open-corpus procedures have imperfect recall.
- `not-observable` — the corpus slice that would have to be searched is not available, so the family
  returns no evidence in either direction.

This separation lets the Olmo3 arms
compare the temporal proxy descriptively with confirmed-positive corpus evidence without treating either
publication time or search non-detection as perfect ground truth.

One asymmetry specific to the Llama-3.1-8B arm: LLMLagBench detects its knowledge-drop changepoint at
2023-03, nine months before the declared 2023-12 cutoff (§4.1). We use the declared, later date for the
primary label because a knowledge drop is evidence of thin coverage, not proof of non-exposure. Items
from 2023-04-01 through 2023-12-31 are flagged `boundary_ambiguous`: they are `possible-exposure` in the
declared-boundary primary run and `clean-by-model-cutoff` in the detected-boundary sensitivity run. Since
LCB collection begins in 2023-05, the sensitivity run leaves no LCB `possible-exposure` items for this
arm. Its secondary suspect conditions are unaffected: HumanEval and the original MBPP problem statements
date to 2021, although MBPP+'s augmented tests were released later.

For Llama's own LCB Q1b/Q2 contrasts, the detected-boundary sensitivity result is pre-specified as
**not estimable: zero possible-exposure items**, not as a second numerical estimate to compare with
the primary result. The pre-sensitivity LCB score distribution is also absent. Report these empty
groups explicitly. For any exploratory pooled sensitivity analysis, exclude the entire Llama arm
from that contrast (including its controls), compare the remaining models under both label rules,
and report the altered model population; this cannot establish robustness of Llama's own estimate.
Boundary sensitivity needs no additional generation or scoring.

The rule generalizes beyond this arm: **any arm whose cutoff evidence is bracketed by competing bounds is
re-analyzed under those bounds**. Qwen2.5 currently has only its release-date bound. The two Olmo3 cards
declare the same date and no competing bound is available for that arm, so there is no second label rule to
re-analyze under; that is an absence of a bracketing bound, not independent corroboration of the declared
date.

Two properties of that declaration bear directly on the shared control. First, the cards state
`Date cutoff: Dec. 2024` but do not say whether the date covers the post-training stages as well as
pretraining, and this design treats post-training as an exposure path (§4.1). Second, both Olmo3 Instruct
checkpoints were published in late 2025, after every item in the shared-control window, so the release-date
reasoning applied to Qwen2.5 yields no constraint here. The 182 `shared-clean-control` items are therefore
temporally clean for these two arms only insofar as that one declared date covers every training stage. We
accordingly report the Olmo3 corpus-reference search results (§5, step 5) for the 182 control items
separately from the suspect cells, so a confirmed match inside the control window is visible rather than
absorbed into an aggregate; a search that finds no match still does not count as a verified negative, per
the corpus-evidence axis above.

As a purely descriptive check, we also compare the ambiguous-window items'
full-precision detector-score distribution against the post-primary-boundary and pre-sensitivity-boundary
distributions where those groups exist; unavailable comparisons are marked not estimable. This comparison is reported separately and is **never fed back into
Q1b's labels** — doing so would let the detectors under evaluation adjudicate their own ground truth.

Every HumanEval and MBPP+ item sits on the possible-exposure side, so neither benchmark contains a control
group of its own. At their native sample sizes they support **exploratory Q1a** only — the paired
within-item precision shift, which needs no exposure label — and not **Q1b**, whose proxy-AUC contrast
requires two labelled groups inside the same benchmark (§4.5.2). For **Q2** they are separate exploratory
contrasts, never pooled with each other. HumanEval's 164-item ceiling limits its secondary Q2 contrast to a
15.5-point best case regardless of how much shared-control data is collected (§4.5.3). The primary Q2 cell
is the LiveCodeBench `possible-exposure` versus `shared-clean-control` contrast.

### 4.3 Quantization axis

**bf16 baseline** → **BNB int8** → **BNB-nf4** → **AWQ-int4**

Double quantization is excluded to keep the four-condition scope fixed; we do not assume that it
cannot affect accuracy or detector scores. AWQ, implemented through llm-compressor, supplies a
calibration-based comparison to BNB. This supports comparisons among these configurations, not an
unrestricted claim about quantization in general. The nf4 contrast is pre-selected as a plausible
stress case; the long-context evidence and calibration distinction do not establish that it has the
largest code-task or detector effect. Engineering tests may exercise it first for operational reasons
without using outputs to select the study arm (§4.6).

**Condition settings.** The four rungs are fixed as follows; "int4" means 4-bit weights with 16-bit
activations in both int4 conditions, and activations are not quantized anywhere in the ladder.

| Condition | Implementation | Key settings | Modules left unquantized |
|---|---|---|---|
| bf16 baseline | transformers, `dtype=bfloat16`, `device_map="auto"` | no quantization | all |
| BNB int8 | bitsandbytes at load time, `load_in_8bit=True` | LLM.int8() mixed-precision decomposition; the outlier threshold (`llm_int8_threshold`) is left at the library default | the library's default skip list (language-model head) |
| BNB-nf4 | bitsandbytes at load time, `load_in_4bit=True` | `quant_type="nf4"`, compute dtype bf16, double quantization **off** (`bnb_4bit_use_double_quant=False`); block size left at the library default | the same default skip list |
| AWQ-int4 | llm-compressor one-shot offline, checkpoint then loaded through transformers/compressed-tensors | `AWQModifier(duo_scaling="both")` with `QuantizationModifier(scheme="W4A16_ASYM", targets=["Linear"])`: **asymmetric** 4-bit weights over every `nn.Linear` layer; group size left at the scheme's default | `lm_head`, excluded explicitly |

Settings described as left at a library default are not overridden by us; the resolved value is recorded in
the run manifest, so reproduction reads the record rather than this sentence. Excluding the language-model
head matters for this study specifically, because the probability detectors of §4.4 score logits produced by
that layer.

All four conditions run on **one inference stack** — PyTorch with Hugging Face transformers `generate()` —
with bitsandbytes quantizing at load time for the two BNB rungs and compressed-tensors loading the offline
AWQ checkpoint. No second runtime is introduced, so a precision contrast is never also a change of inference
engine. Kernel-level differences between the bitsandbytes and AWQ paths do remain inside the AWQ-versus-BNB
comparison. Library versions are pinned and recorded with each run (torch 2.13.0+cu130, transformers 5.14.1,
bitsandbytes 0.50.1, accelerate 1.14.0, llm-compressor 0.13.0, compressed-tensors 0.18.0), together with the
GPU driver and CUDA versions.

Before the main run, freeze one AWQ calibration artifact per model and record its dataset revision,
selected-row hashes, seed, tokenizer, quantizer recipe and software versions. The current code-calibration
candidate is `flytech/python-codes-25k`, revision `0ed98ff2a76c5d133d8c157b814189a5a17ebd20`,
256 rows shuffled with seed 42, maximum sequence length 512. Calibration reads that dataset's own
`text` column — instruction text plus a fenced Python block — as shipped, with no chat template applied; the chat
comparison set (`HuggingFaceH4/ultrachat_200k`, revision `8049631c405ae6576f93f445c6b8166f76f5505a`) is
rendered through each model's own chat template first, because its rows are message lists. Exactly one
calibration artifact per model enters the main run. Check candidates against all evaluation
prompts and reference solutions before use; record search coverage and exclusions. This overlap check
is a required pre-execution step, not a completed result or proof of semantic non-overlap. Calibration
exposure would be an additional source of benchmark information. Existing code/chat engineering
comparisons do not choose calibration from observed detector or task performance.

### 4.4 Detection signals (for Q1)

- **Peakedness family:** CDD — *Contamination Detection via output Distribution* — the
  output-distribution-peakedness detector characterized in §2.4, introduced by Dong et al. (2024).
  (Expansion and attribution as given in arXiv:2603.03203's abstract and introduction.)
- **Probability family:** perplexity, Min-k% Prob.

**Frozen scoring protocol.** At each model/precision/item, generate one greedy output and n=50
samples at temperature 0.8, with a 512-token generation cap. Generation runs from an
execution-augmented prompt while the probability detectors score the unaugmented benchmark text: for
LiveCodeBench, the generation prompt appends the item's own starter code plus an instruction to complete
it when the item ships starter code, and otherwise an instruction to write a program that reads standard
input and writes standard output, whereas HumanEval+ and MBPP+ prompts are used as shipped. The greedy output and
all 50 samples — and therefore both CDD and pass@1 — come from that augmented prompt; perplexity and
Min-k% teacher-force the problem statement alone, as specified below. The augmentation is identical at
every precision, so it cannot differ across a precision contrast. Every decoding setting is specified
explicitly and the checkpoint's own `generation_config` is not followed: `top_p=1.0`, top-k sampling
disabled, `repetition_penalty=1.0`, and no length penalty or minimum-length constraint. The same settings
apply to the greedy reference output, which differs from the samples only in that sampling is
off — repetition penalty in particular would change the greedy output as well. This is not a redundant
restatement of defaults: shipped configurations differ across the roster (Qwen2.5-32B-Instruct, for
example, ships `temperature 0.7`, `top_p 0.8`, `top_k 20`, `repetition_penalty 1.05`), so inheriting them
would truncate each model's CDD sample distribution differently and make the peakedness scores
incomparable across arms. The seed for each generation is derived as sha256(item_id, sample_id,
temperature) and set immediately before that generation, so a given item and sample index draw the same
seed at every precision; the seed-policy identifier is stored in the run manifest. All precisions are
generated on the single inference stack described in §4.3. For CDD, truncate output token IDs to at most 100 tokens, excluding the prompt. Let l be the
maximum actual length across the truncated greedy output and all 50 samples. The score is
(1/50)Σ I[ED(sample_i,greedy)≤0.05l], using token-level Levenshtein distance. l is not automatically
100: ten-token outputs use a 0.5-edit threshold. This follows arXiv:2603.03203 §3.1 equation (1).
Q1 uses the raw peakedness score, which lies on the discrete grid {0, 1/50, …, 1}, rather than a binary ξ decision.

Perplexity and Min-k% use teacher-forced natural-log probabilities of one fixed text per item, never a
sampled answer and never a reference solution. The scored text is the benchmark's own problem statement:
for LiveCodeBench, the `question_content` field in full — including the worked examples and sample
input/output blocks that the field itself carries, which nearly every `release_v6` item has — with only
what generation adds left out, namely the starter-code block appended to the items that ship starter
code, and the execution-format instruction; for HumanEval+ and MBPP+, the shipped prompt field (signature
and docstring), with the canonical solution excluded. The scored text is therefore identical across
precisions, and the same field is scored for every LiveCodeBench item whether or not that item ships
starter code. That text is placed in a single user message and rendered
with the checkpoint's own chat template, with the assistant generation marker appended. We add no system
message of our own, so a template that inserts a default system string or a date line inserts the same one
at every precision. Only tokens lying wholly inside the benchmark text contribute to the score: template,
special and generation-marker tokens are excluded by character offsets, and a target token with no causal
left context is dropped. Perplexity is exp(−mean log probability). For Q1a and the stored `perplexity` score, use the numerically stable −log(perplexity), i.e. mean log probability; its shift is a different estimand from raw-perplexity shift. Min-k% uses k=20 and
averages the lowest max(1, round(0.2N)) of the N scored token log-probabilities. Its source defines that
set as "the k% of tokens" with the minimum probability and fixes no rule for a non-integer 0.2N
(arXiv:2310.16789 §3 and its Algorithm 1), so the convention is stated here rather than left implicit:
round to nearest, with a floor of one token. Truncating instead — the other natural reading — selects one
token fewer whenever N ≡ 3 or 4 (mod 5) and N ≥ 8 (two tokens versus one at N=8), so the two conventions
give different scores on short items.
Completion-based scores are separate diagnostics. Record target text, token boundaries, truncation,
chat template, tokenizer/checkpoint revisions and decoding settings so precision comparisons use
identical text and scoring rules.

**Pass@1 and partial credit.** pass@1 is scored from the single greedy output at each model/precision/item,
not estimated from the 50 temperature samples, and an item counts as passed only when it passes every test
case. Because the roster is entirely instruction-tuned and queried through a chat template, code is
extracted from the chat-formatted output before execution: the last fenced block is taken when the output
contains any, and the result is then passed through evalplus's own post-processing — its AST-based
`sanitize` anchored on the target entry point for HumanEval+ and MBPP+, and its `code_extract` for
LiveCodeBench, whose stdin-style programs often have no entry point to anchor on. Test sets are the full
ones: HumanEval+ and MBPP+ are scored on both the base and the plus inputs, at the versions the pinned
evalplus release resolves, with each dataset's own version hash stored per run; LiveCodeBench is scored on
its public and private test cases together, from `release_v6` of `livecodebench/code_generation_lite` at the
recorded repository revision. Partial credit is the fraction of those test cases passed. Its denominator is
the full test-case count in both families, so a case the harness never reached after an earlier crash or
timeout counts as failed rather than being dropped from the denominator. Each LiveCodeBench test case runs
in a separate subprocess under a 6-second wall-clock timeout; HumanEval+/MBPP+ execution uses evalplus's own
containment with a per-test time limit derived from the reference solution's runtime. The execution
environment is not a hardened sandbox: the LiveCodeBench path has no network isolation and no memory cap
beyond the operating system's, which we record as an execution-environment limitation rather than treat as
isolation. Finally, the 512-token generation cap is shorter than LiveCodeBench's official runner default, so
we record per item whether generation stopped at the cap and report the truncated-generation rate by
precision: a truncation rate that differs across precisions would confound a pass@1 shift with a
length-cap artifact.

CDD's 51 generations dominate generation cost and can be shared with pass@1/partial-credit scoring;
the two probability detectors need only fixed-text forward scoring. CDD sample count affects score
resolution and uncertainty. Any operational change to n must occur before study outcomes are inspected,
be documented and frozen, and be reflected consistently in the scoring formula and manifest (§4.6).

**Threshold handling (ξ).** The original CDD paper fixes a detection threshold ξ=0.01, calibrated on 7B
models (as reported in arXiv:2603.03203 §3.1); arXiv:2603.03203 itself re-selects ξ per condition via
Youden-index maximization on its own evaluation set and small models,
and explicitly notes this "gives CDD every advantage" — i.e., it is an optimistic, oracle-selected
threshold. Because **Q1b's primary metric is AUC, which is threshold-independent**, ξ recalibration is
*not* required for Q1b. It is only relevant if CDD point-accuracy is reported as a secondary descriptive
statistic, in which case ξ must **not** be re-selected on the evaluation set per condition (which would
reproduce arXiv:2603.03203's evaluation-set Youden selection, not the original paper's fixed
threshold); instead, apply either an identical, pre-fixed
threshold across all precision conditions, or a threshold calibrated on a held-out split.

### 4.5 Statistical design

#### 4.5.1 Q1a — paired detector-score shift

The confirmatory statistic is the mean within-item nf4−bf16 score difference, tested with a two-sided
paired t-test (§4.5.6). Cohen's d here is d_z, the mean difference divided by the standard deviation of
item differences. Exploratory multi-model fits do not replace these three confirmatory tests
(C1–C3 in §4.5.6).

| Effect size (d_z) | Approximate items (80% power, α=0.05, paired normal approximation) |
|---|---|
| 0.3 | 87 |
| 0.2 | 196 |

The formula is n=(z_0.975+z_0.8)²/d_z², with multiplier 2.8016. Values above are rounded planning
approximations: the unrounded roots are 87.210 and 196.222, so strict integer ceilings under this
normal approximation are 88 and 197. Exact finite-sample t-test power is a different calculation.
These d values are illustrative, not predictions. Small-model CDD results do not establish a step
function or a stable effect size for our models, nor guarantee more stable probability-detector effects.
The frozen main run uses all planned items; engineering validation does not resize it (§4.6).

#### 4.5.2 Q1b — detector-family proxy-AUC comparison and construct-validity limits

AUC separability of `possible-exposure` vs. `shared-clean-control` items, per detector, per precision;
compared as **paired** AUCs (both computed from the same model–item set) since the same items are scored
by both detectors. On Olmo3, the independent corpus-status axis additionally supports a confirmed-positive
descriptive validation analysis; it does not replace the temporal labels in the other arms.

| Items per label group | SE(AUC) | Detectable ΔAUC, r=0 | r=0.8 | r=0.9 |
|---|---|---|---|---|
| 164 (328 items in total) | 0.029 | 0.114 | **0.051** | 0.036 |
| 300 (600 in total) | 0.021 | 0.084 | 0.038 | 0.027 |
| 542 (1,084 in total; hypothetical reference only) | 0.016 | 0.063 | 0.028 | 0.020 |
| 1,000 (2,000 in total) | 0.012 | 0.046 | 0.021 | 0.015 |

Every row's *n* is the count **per label group**, `possible-exposure` and `shared-clean-control` alike,
so the AUC is computed on twice that many items. This Hanley–McNeil planning example assumes AUC A=0.70,
equal groups of n items, and normal approximations: Q₁=A/(2−A), Q₂=2A²/(1+A), and
SE²=[A(1−A)+(n−1)(Q₁−A²)+(n−1)(Q₂−A²)]/n².
The detectable difference is 2.8016√[2(1−r)]SE, where r is the correlation between the **two AUC
estimators**, not between individual detector scores. At n=164 per group, SE=0.028732 and the
r=0.8 limit is 0.050910. The 0.05 target gives n=170.005, approximately 170 but a strict ceiling of
171 under this approximation. HumanEval alone supplies neither two groups of 164 nor the required
LCB proxy contrast; HumanEval and MBPP+ remain separate. Actual LCB groups are unequal (e.g. 690/182
for Qwen), so this table is illustrative, not their achieved precision. A two-AUC difference calculation
does not power the rank-reversal test C4, which combines six correlated AUC estimates (§4.5.6).

**Construct validity is the more serious threat to Q1b.** Q1b's estimand is explicitly the AUC for the
stored temporal proxy labels, so no unknown error rate is required to compute that estimand. If one tries
to reinterpret it as AUC against unobserved true contamination labels, however, proxy-label error can
attenuate the corresponding true-label AUC difference. Under the table's explicitly simplified
sensitivity model — balanced true classes (prevalence 0.5), with symmetric, nondifferential label flips at a hypothetical rate *e* — the relation is
approximately ΔAUC_observed ≈ (1 − 2e) × ΔAUC_true:

| Hypothetical proxy-label error rate *e* | Observed ΔAUC (true = 0.050) | Items needed per label group (r=0.8) |
|---|---|---|
| 0% (proxy is exact) | 0.050 | 170 |
| 10% | 0.040 | 287 |
| 20% | 0.030 | 541 |
| 30% | 0.020 | 1,268 |

Each row uses A_obs=0.5+(1−2e)(0.70−0.5) in the same SE formula, attenuates ΔAUC=0.050 by
(1−2e), and solves at r=0.8. Rounded roots are 170/287/541/1,268; strict integer ceilings are
171/288/542/1,268. With unequal true prevalence, symmetric forward label flips alone do not imply
(1−2e) attenuation: the factor is Pr(Y=1|observed positive)−Pr(Y=1|observed negative).
For example, prevalence 0.2 and e=0.2 give 0.4412, not 0.6. This balanced hypothetical model is not
asserted for the actual temporal labels.

At e=20%, a true-label interpretation would require ≈541 items — the same order as Q2's paired-model
requirement (≈555). No arm is assigned to one of these rows empirically: even for Olmo3,
`no-match-found` is not proof of non-exposure, so the corpus search cannot identify a binary error rate,
false-positive rate, or false-negative rate for the temporal proxy. The table is therefore a
construct-validity sensitivity analysis only, not a correction or a data-dependent sizing input. Olmo3 corpus
results are reported separately as positive matches and searched non-matches, without changing Q1b's
proxy-label estimand. Q1a does not use an exposure label and is unaffected by this limitation.

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

For the primary LCB contrast, the shared-control ceiling gives an additional hard bound under the same
conservative unpaired calculation. With 182 controls per precision and an infinite suspect cell, the
minimum detectable interaction is 14.7pp; with the largest possible 873-item suspect envelope it is
16.1pp. Because suspect membership is model-specific, 16.1pp is an optimistic lower bound rather than a
guaranteed design point. The HumanEval-versus-LCB comparison is a secondary-contrast power diagnostic
only: 164 HumanEval items and infinite controls imply 15.5pp.

**Sample-size decomposition (do not conflate these — they answer different questions):**

| Scenario | *n* needed (10pp) | Source of the reduction |
|---|---|---|
| Unpaired, p=0.5 both conditions | **785** | — (independent-cell raw-pp benchmark; **use this for planning**) |
| Unpaired, actual base rates 0.85/0.35 | 557 | Base rate alone: −29% (extreme base rates shrink binomial variance) |
| Paired (item difficulty SD=1.5 model), p=0.5 | 555 (implied r = 0.293) | Pairing alone: −29% |
| Paired (SD=1.5 model), actual base rates | **≈412** | Both effects combined: −47% |

*All four rows share one convention: binomial variance is evaluated at the null (no-drop) base rates,
and any pairing discount uses the σ=1.5 item model's implied cross-precision correlation at that null
(r by numerical integration; 0.293 at p=0.5, 0.222 at 0.85 and 0.283 at 0.35). Row 1 is p=0.5 and
unpaired; row 2 changes only the base rates; row 3 changes only the pairing; row 4 changes both at once
and is a separate joint computation, not the product of rows 2 and 3. The base rates 0.85 and 0.35 are
marginal means over the σ=1.5 difficulty distribution, not per-item logits. That rows 2 and 3 land within
two items of each other (557 vs. 555) is coincidence — different mechanisms. Evaluating row 4's variance
at a non-null drop instead would give 415 (β=0.25) to 426 (β=1.0), so that figure is convention-dependent
in a way the planning benchmark is not.*

785 and ≈412 differ by nearly 2×. **785 remains the adopted planning benchmark**, under independent
Bernoulli cells, p=0.5, equal cell sizes, a raw-percentage-point contrast and a normal approximation.
It is not an assumption-free upper bound or a power calculation for the paired log-odds GLMM.
The paired reductions require their stated base-rate and correlation assumptions; multiplying reductions
from incompatible scales is invalid. Main-study correlations are reported descriptively and do not
trigger outcome-dependent resizing. Engineering validation does not estimate these quantities.

**The base-rate confound.** HumanEval (bf16 pass@1 ≈ 0.85) and LiveCodeBench-post (≈0.35) have very
different baseline accuracies. These two figures are illustrative values for a Qwen-class instruction-tuned
model; the actual base rates differ by model — Olmo3's in particular should not be assumed to match
Qwen2.5's — and are estimated per model from the frozen main-study data. The sign and magnitude below depend on these illustrative base rates and a shared normal item-difficulty distribution with SD=1.5; neither baseline separation nor its direction is established for every model.
Both 0.85 and 0.35 enter as **marginal means over that SD=1.5 difficulty distribution**, the same
convention as the decomposition table above, not as a single per-item logit: reading them as μ=logit(p)
instead changes every row of the table below.
On the raw percentage-point scale, this difference alone produces a
**spurious interaction** even when the true, item-conditional quantization effect (in log-odds) is
*identical* across both conditions:

| Item-conditional log-odds drop β | HumanEval drop | LCB-post drop | **Spurious %p interaction** |
|---|---|---|---|
| 0.25 | 2.6pp | 4.0pp | **−1.4pp** |
| 0.50 | 5.6pp | 7.7pp | **−2.2pp** |
| 0.75 | 8.8pp | 11.2pp | **−2.5pp** |
| 1.00 | 12.3pp | 14.5pp | **−2.2pp** |

All rows are negative for this particular 0.85/0.35 example. The sign is not universal: with the
same SD=1.5 difficulty model and β=0.5, baselines 0.60/0.10 produce drops of 8.72/3.15pp, a **positive
5.57pp** interaction. These are model calculations, not experimental observations.

**Mitigation:** estimate a conditional log-odds interaction with explicit coding (§3.1). Difficulty
stratification remains a required diagnostic: report quantization effects across difficulty strata and
overlap between exposure groups. A nonsignificant heterogeneity test cannot confirm constant odds ratios.
Insufficient overlap or systematic differences limit interpretation; any additional matching or altered
model is exploratory, not an outcome-dependent replacement of the planned analysis.
For the illustrated 0.85/0.35, SD=1.5 model, marginal aggregation produces residual log-odds interaction
of about 0.02305 at β=0.75 (and 0.02047 at β=0.5), despite a zero conditional interaction. This is a worked example, not a universal
bias bound. A correctly specified conditional model addresses this aggregation artifact, but does not
guarantee removal of omitted difficulty effects, distribution misspecification, or exposure confounding.

#### 4.5.4 Reconciling the numbers

**785** is the adopted independent-cell, p=0.5 raw-pp normal-approximation benchmark. **≈412** is
predicted only under the additional difficulty SD=1.5, illustrative marginal base-rate and
conditional-independence assumptions (§4.5.3). Neither is direct power for the planned log-odds model. The benchmark remains
fixed; validation observations cannot lower it or establish retrospectively improved power.

#### 4.5.5 Statistical model

```
correct ~ precision * exposure_proxy + (1 | item) + (1 | model)
```

This is an exploratory pooled logistic random-intercept working model. Fit one quantized level versus
bf16 at a time, coding precision and exposure as Q and E in §3.1; labels are constant across precision
within each model–item pair, not necessarily across models. Report model-specific contrasts first.
Random intercepts represent baseline item/model variation and repeated-item dependence. They do not
represent model-specific quantization slopes, model-specific interactions, or all item-by-model
dependence. In particular the working model carries **no item-specific quantization slope**: an
item × precision random term is omitted, so genuine item-to-item variation in the quantization effect is
absorbed into residual noise and the pooled fit's β_QE interval can be narrower than that variation
warrants. We have not simulated how large that understatement is, so it is listed as a limitation of the
pooled working model rather than quantified. Report heterogeneity and difficulty diagnostics; five model arms do not support a strong
population-wide claim about architectures or sizes. Per-model fits omit the model random intercept.

**The interval for β_QE is not taken from a mean-field variational Bayes posterior SD.** A mean-field
approximation factorizes the posterior across coefficients and discards their correlations. In a 2×2
treatment-coded design the resulting posterior SD is several times smaller than the sampling variability
of the estimate itself, so mean ±1.96 posterior SD does not carry its nominal coverage and is not
reported as an interval. The reported interval comes instead from an item-stratified conditional
logistic fit, run separately per model, in which item difficulty is conditioned out rather than modelled:
the regressors are Q and Q×E under §3.1's coding, the item intercepts drop out of the conditional
likelihood, and the interval is the Wald interval for β_QE. Two alternatives may replace it — MCMC from
the full posterior, or a Laplace approximation whose covariance inverts the full joint Hessian over
fixed effects and random intercepts together — provided the same coverage check below is passed first.
statsmodels `BinomialBayesMixedGLM.fit_vb` may still be used for point estimates and variance
components, but not for interval width.

**Verify interval coverage on synthetic data before the main run.** Generate data from the design's own
shape — one model's item counts (e.g. 690 `possible-exposure` and 182 `shared-clean-control`), a normal
item-difficulty distribution, and a known β_QE — and estimate the achieved coverage of each candidate
interval method over enough replications to separate it from the nominal 95%. Record the generating
parameters, the number of replications, and the achieved coverage of every method examined in the
analysis manifest, alongside the priors, coding, convergence diagnostics and library versions. That check
fixes which method supplies the reported interval; the main-study output does not (§5, step 7). Report
β_QE and J=−β_QE; negating an interval also reverses its endpoints. Q2 remains secondary and
interval-focused. This working model and its uncertainty require diagnostic review before any
substantive interpretation.

#### 4.5.6 Multiplicity and confirmatory scope

The **four confirmatory tests are restricted to Qwen2.5-32B-Instruct**, the Primary model in §4.1,
and the bf16→BNB-nf4 contrast. All other models and quantization contrasts are exploratory.

- **C1–C3 (Q1a):** one two-sided paired t-test per detector (perplexity, Min-k% Prob, CDD), on all
  1,055 LCB release_v6 items, including the 183 intermediate-date items. The estimand is the mean
  within-item nf4−bf16 score difference; no exposure label is used. Each test runs on complete cases:
  an item enters only when all six scores (three detectors × two precisions) are present and finite, and
  an item missing any of them is dropped from all three tests together so that C1–C3 stay on one common
  item set; the number dropped is reported. If that leaves fewer than two complete pairs, or the item
  differences have zero variance, the slot is reported as not estimable with p=1 and kept in the family,
  under the same rule stated for C4 below.
- **C4 (Q1b):** a literal reversal of the probability-family versus CDD AUC ranking. Use only the
  690 `possible-exposure` and 182 `shared-clean-control` items. At each precision p, define
  g_p=[AUC_perplexity,p+AUC_Min-k,p]/2−AUC_CDD,p. This is an equal-weight mean of two AUCs,
  not an AUC of pooled raw scores; report both component AUCs too. Larger scores always indicate
  more possible exposure (negative log perplexity, unmodified Min-k log-probability and CDD).
  A reversal requires g_bf16>0 and g_nf4<0, or g_bf16<0 and g_nf4>0. A change in gap alone is
  insufficient, and observed sign changes alone are descriptive rather than confirmatory evidence.

C1–C3 test each detector's own mean shift, not differences between detector effects. A significant
shift for one detector and a nonsignificant shift for another do not establish differential modulation;
the raw score scales are also not directly comparable. C4 addresses reversal of the pre-specified mean
probability-family AUC relative to CDD, not reversal of each probability detector individually. Any
additional direct comparison of score shifts or AUC-gap changes is exploratory and does not expand
the four-test confirmatory family.

For C4, estimate the joint [DeLong covariance](https://pubmed.ncbi.nlm.nih.gov/3203132/) of the six AUC estimates on the same items and obtain
each gap's standard error by the corresponding linear contrast. Let p⁺_b and p⁻_b be one-sided normal
Wald p-values for g_bf16>0 and <0, and p⁺_q and p⁻_q those for g_nf4. Use
p_forward=max(p⁺_b,p⁻_q), p_reverse=max(p⁻_b,p⁺_q), and
**p_C4=min(1,2 min(p_forward,p_reverse))**. Each direction is an intersection–union test; the factor
two corrects choosing either direction. These p-values rely on the asymptotic AUC approximation.
An empty label group, missing required scores, or a zero/undefined variance of either gap is reported as not estimable;
retain the slot with p=1 for multiplicity accounting rather than removing it or selecting another test.
Singularity of the full six-AUC covariance alone does not invalidate these scalar contrasts; no matrix inverse is required. A constant CDD score alone is not a reason to remove C4 when both gap variances remain positive.

Apply Holm at familywise α=0.05 to these four p-values. Other contrasts, HumanEval/MBPP+, Q2 and
boundary sensitivity are exploratory, with intervals and no confirmatory significance claims; Q2 uses
the coverage-verified intervals specified in §4.5.5. Q1a normal-approximation sizing at α/4 uses multiplier
3.339: d_z=0.3 requires ≈124 items and d_z=0.2 ≈279. That figure does not size C4, whose precision
depends on both signed gaps and the joint covariance. An illustrative C4 calculation, under assumptions
that are stated rather than established: all six AUCs equal 0.70 on 690 vs. 182 items, giving
Hanley–McNeil SE 0.0198 each; the two probability AUCs correlate at 0.8–0.9 and each probability AUC
correlates with CDD at 0.0–0.5; the intersection–union directions are treated as independent, so each
needs power √0.80. SE(g_p) is then 0.019–0.028, and at Holm's first step (familywise α/4=0.0125, i.e.
a one-sided 0.00625 per direction because p_C4 already carries the factor two) 80% power requires
|g_bf16| and |g_nf4| of about **0.07–0.10** each, or about 0.06–0.09 at Holm's last step. A detectable
reversal therefore needs the probability-minus-CDD AUC gap to swing by roughly 0.14–0.21 between bf16
and nf4 — a much larger quantity than §4.5.2's 0.051 two-AUC detection limit, which sizes a different
comparison. The family, score orientation and item sets are fixed before main-study outcomes.
Engineering validation cannot change them. The analysis code for C1–C4, for the §4.5.5 intervals, and
for the interval-coverage check specified there must be implemented and validated on synthetic data
before the main run (§5, step 7).

### 4.6 Engineering validation boundary

The local dry run and bounded real-hardware smoke tests are **engineering validation only**. They verify
model loading, quantization compatibility, item-schema integrity, finite teacher-forced log-probabilities,
sandboxed execution, deterministic output paths, memory use, and throughput. Synthetic and smoke-test
outputs are stored in a validation-only namespace and are never used as manuscript evidence.

Validation outputs are not aggregated into detector effect sizes, proxy AUCs, base rates, cross-precision
correlations, power estimates, or detector-ranking decisions. In particular, CDD is not screened through a
data-dependent gate. Two kinds of change are permitted during validation: an operational change motivated
by a runtime or memory failure (batch size, execution chunking, and the like), and a correction to a defect
in the scoring harness itself — a code-extraction rule that drops valid completions, a sandbox invocation
that fails independently of the model's answer, a misaligned teacher-forced log-probability span. Both are
permitted **only before detector or task-performance outcomes are inspected**, and both must be recorded
with what was observed, what was changed, and when. The changed configuration is then frozen before the
main run begins. No change of either kind is permitted after detector scores, proxy AUCs, or pass rates
have been looked at, and a defect correction may not be selected by comparing the outcomes it produces.

### 4.7 Fixed study analysis without a data-dependent pilot

This design has no scientific pilot. The confirmatory family C1–C4 is fixed before the main run, and CDD is
not removed or demoted based on validation observations. If CDD is near chance in the main study, that
estimate and its confidence interval are reported as a substantive result rather than used to redefine the
analysis family.

The main run uses the frozen model roster, item definitions, temporal boundaries, scoring procedures, and
all planned available items. Detector-score shifts, proxy AUCs, base rates, and cross-precision correlations
are first treated as study quantities in that run. They may characterize achieved precision and support
the pre-specified sensitivity analyses, but they do not trigger post-validation resizing or confirmatory
reclassification. Olmo3 corpus matches remain a separate source of model-local confirmed-positive evidence.

---

## 5. Execution Plan

1. **Build the continuous-scoring pipeline first** (partial test-case pass rate + token log-probability).
   At the available item counts, §4.5.3 does not support Q2 as the primary question.
   Continuous scoring is required for Q1; partial test credit is exploratory.
2. **Build the detector-scoring pipeline** (CDD, perplexity, Min-k% Prob per item, per precision) —
   required for Q1. Budget CDD's per-item multi-sample requirement (§4.4) into the generation-cost
   estimate; design steps 1 and 2 to share underlying generations wherever possible.
3. **Materialize and count the model–item temporal labels** (§4.2), rather than splitting LCB once globally.
   For every model–item pair store `publication_date`, `primary_first_post_date`,
   `sensitivity_first_post_date` (if any), `possible-exposure` / `clean-by-model-cutoff`,
   `shared-clean-control`, and `boundary_ambiguous`. Under `release_v6`, the common 2025-01-01 split still
   gives the availability envelope pre 873 / shared control 182 / total 1,055, but the suspect count is
   recomputed per arm and is not reported as 873 for every model. The ≥1,000 target is therefore unmet;
   **Q2 remains a secondary, interval-focused analysis**.
4. **Verify and freeze the model-level temporal bounds** rather than trusting one global date.
   Llama-3.1-8B uses first-post dates 2024-01-01 (declared; primary) and 2023-04-01 (detected;
   sensitivity). Qwen2.5 has no unambiguous official cutoff declaration, so its first-post date is fixed
   at 2024-09-20, the day after the official release. Both final Olmo Instruct model cards state
   `Date cutoff: Dec. 2024`, making 2025-01-01 their first post-boundary date. Direct searches of the
   public pretraining and post-training corpora populate the separate corpus-status axis, not the temporal
   cutoff label.
5. **Search for residual-contamination evidence via TRACER (arXiv:2605.24079), against the released Olmo3
   pretraining and post-training corpora** — the only model-training pipeline in the design open
   enough to run it on: TRACER is defined over a pair (post-training dataset, evaluation benchmark) and
   classifies each candidate task pair, and Qwen2.5's and Llama-3.1's corpora are closed (§4.5.2). Its
   source validates it only in that setting — three benchmarks against three code post-training corpora
   (CodeAlpaca-20K, Evol-CodeAlpaca-V1, Magicoder-OSS-Instruct-75K), with its triage thresholds tuned on
   a development split of that annotated benchmark. **Running it against a pretraining corpus is our
   extension beyond the source's validated setting**, and its accuracy there is unestablished: reported
   matches are candidate positives to adjudicate, not calibrated detections. This is
   descriptive construct-validity evidence for Q1b but is not an error-rate estimate or a prerequisite for
   computing the proxy AUC; it can proceed in parallel with step 6. TRACER has no
   confirmed public code release; we reimplement it from the paper's own
   specification (its appendix publishes the prompts for all three LLM stages, the embedding model, and
   the triage thresholds), with a retrieval pre-stage (n-gram/BM25 top-k candidates per benchmark item)
   in front, since exhaustive pairwise comparison against a pretraining-scale corpus is infeasible.
   **For the Olmo3 arm, also derive operational corpus-reference statuses** from the released training data — the
   pretraining corpus *and* the post-training (instruction-tuning) sets, both public for Olmo3, since the
   instruct checkpoints (§4.1) can absorb benchmark items at either stage. We run all three open-data
   detection families in arXiv:2404.00699's taxonomy rather than choosing one, because each family
   accepts a different kind of match and no one family's count stands in for the others. Whatever spread
   appears between them is reported as a descriptive result; no size of spread is predicted here:

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
     is therefore not pre-suppressed and supplies a genuine lower bound on confirmed lexical matches. The
     opposite risk — a near-zero result that just re-measures the corpus builders' own filter rather than
     the absence of exposure — applies if family (i) is instead run against the midtraining or
     post-training slices, which *are* filtered against these benchmarks; treating their near-zero match
     count as verified non-exposure would create a spuriously optimistic conclusion.
   - **(ii) Surface and structural program matching.** Use edit distance and AST-based similarity
     to retrieve candidate program matches, informed by arXiv:2403.04811. AST similarity is structural
     evidence, not proof of semantic equivalence. Validate thresholds and adjudicate candidate matches
     for the current corpus and benchmark; transferability and positive rates are not assumed.

   - **(iii) Paraphrase detection** — following the retrieval-then-LLM-judge design of arXiv:2311.04850
     (embedding retrieval of top-*k* candidates, then a strong-model judgment on semantic equivalence),
     applied to the **pretraining and post-training sets alike**. That work applies its method to
     pre-training and fine-tuning datasets and states its overlap finding for pretraining corpora — 8–18%
     of HumanEval found in RedPajama-Data-1T and StarCoder-Data — so we do not assume in advance that
     rephrased benchmark items concentrate in either stage.

   Report `confirmed-match`, `no-match-found`, and `not-observable` counts from each family separately.
   No operative *e* is selected. The spread between (i) and (ii)–(iii) is reported as a descriptive
   result: it quantifies how many additional positive matches are found beyond lexical matching. §2.4's
   finding in arXiv:2602.12413 concerns 77.5% of benchmark CodeForces problems having at least one
   judged semantic match among top-100 retrieved candidates. Its denominator is benchmark problems,
   not all corpus records. It motivates broader search but does not predict our lexical-versus-semantic
   match gap or make a small gap an unexpected finding.

   Compare both the model–item temporal proxy and the TRACER reimplementation's output with confirmed
   positive matches descriptively. Searched non-matches remain unlabeled for true exposure, so this step
   does not yield *e*, a false-positive rate, or a false-negative rate. The evidence is specific to Olmo3
   and does not alter the proxy labels of the closed-corpus arms.
6. **Engineering validation only** (§4.6): run the local synthetic dry run and bounded H100 smoke tests.
   Verify model loading, quantization compatibility, schemas, finite log-probabilities, sandbox behavior,
   memory, and throughput. Store these outputs separately and do not compute or inspect study effect sizes,
   AUCs, pass rates, detector rankings, or power from them.
7. **Implement and synthetically validate the analysis, then freeze the operational configuration.**
   Write the analysis code before any study observation exists: the C1–C3 paired t-tests, the C4 six-AUC
   DeLong covariance, gap contrasts, intersection–union p-value and Holm correction of §4.5.6, and the
   per-model interval procedure of §4.5.5. Validate it on synthetic data with known generating
   parameters — recovery of the planted effects for C1–C4, and achieved interval coverage against the
   nominal 95% for every candidate interval method, with the replication count and achieved coverage
   recorded. That check, not the main-study output, fixes which interval method is reported.
   Then freeze the operational configuration. Hardware or runtime failures observed before outcome
   inspection may justify changes such as batch size or execution chunking, as may a defect in the
   scoring harness itself (§4.6); record those changes, then freeze the model, item, scoring, and
   analysis configuration before generating study observations. Validation data do not change C1–C4,
   the planned item set, or detector priority.
8. **Full run — the only source of study data.** Store item-level raw data for every condition: pass@1, partial credit, token
   log-probability, and all three detector scores. Aggregate-only storage would foreclose the paired and
   mixed-effects analyses this design depends on.
9. **Analysis — run the code frozen in step 7, without modifying it.**
   - *Q1a/Q1b:* run the fixed tests on the fixed item sets, with the AUC orientation and C4 reversal
     criterion of §4.5.6; report exploratory model/precision contrasts separately.
   - *Q2:* report explicitly coded conditional log-odds β_QE and drop contrast J=−β_QE with the
     coverage-verified intervals and working-model limitations in §4.5.5. Difficulty diagnostics
     assess assumptions; they cannot certify them by a nonsignificant result.
   - *Boundary sensitivity:* apply §4.2's fixed labels and report empty groups as not estimable,
     especially Llama's detected-boundary LCB contrast. Descriptive detector comparisons never alter labels.

---

## 6. Threats to Validity

- **The exposure axis is observational, not causal.** Contamination status cannot be randomly assigned to
  off-the-shelf, already-pretrained 7B–32.5B models; it is an observed covariate, not a treatment.
  Exposure-related results are therefore reported as associations, following arXiv:2501.18771's
  causally-identified design as an aspirational reference this study cannot replicate at this model scale
  (§2.3). The precision axis is different in kind: each precision contrast changes only the numeric format
  of one frozen checkpoint, which is a manipulation we control, subject to the inference-implementation
  caveat below. The observational limit therefore attaches to exposure, and it reaches any result that
  crosses the two axes.
- **Declared training-cutoff dates may be wrong, and cutoff evidence quality is heterogeneous across
  arms.** Only partly addressed. The structural part is §4.2's boundary rule: suspect labels use
  arm-specific bounds, the shared control is restricted to dates after every primary bound, and the per-arm
  evidence tier is reported alongside results. The behavioral part is far narrower than that rule. The
  LLMLagBench diagnostic (§5, step 4) is applied to the Llama-3.1-8B arm only, and this paper treats it as
  a knowledge-boundary diagnostic rather than verification of a training-data cutoff (§4.1); it does not
  reach the arms that set the shared control's start date. That date comes from the two Olmo3 model cards'
  single declared `Date cutoff: Dec. 2024`, with no competing bound available to bracket it and therefore
  no boundary sensitivity run for that arm (§4.2). If the declaration is wrong, or does not cover the
  post-training stages whose scope it leaves unstated, the 182 `shared-clean-control` items are not
  temporally clean for the two Olmo3 arms, and the control side of their Q1b and Q2 contrasts is affected.
  Reporting the Olmo3 corpus-search results for those 182 items separately from the suspect cells (§4.2;
  §5, step 5) is the only direct check available, and it can confirm a match without ever establishing that
  none exists. Residual uncertainty within each tier remains a limitation on the model–item temporal
  labels' precision.
- **"Filtered by date" does not guarantee "uncontaminated."** arXiv:2602.12413 and arXiv:2311.04850
  document that semantic duplication and paraphrase evade n-gram and string-matching decontamination; by
  the same logic, a publication-date filter cannot exclude them either — that extension to date filtering
  is our inference, not a result those papers report. We do not
  claim `shared-clean-control` is clean ground truth; it is a temporal label. On the Olmo3 arms the
  direct corpus search can supply confirmed positive matches as descriptive evidence for §4.5.2, and the
  TRACER reimplementation's output is something to compare against those confirmed positives rather than
  a source of confirmed positives itself; both are planned in §5, step 5 and neither has been run. All
  arms retain temporal
  proxies and an unidentified error rate represented only through the hypothetical sensitivity model.
- **Extrapolation risk from arXiv:2603.03203, on two independent dimensions.** *Scale:* that paper's
  findings are established at 70M–410M, roughly 1.23–2.67 orders of magnitude (about 17–464×) below this
  design's 7B–32.5B range, and the paper itself disclaims extrapolation. *Mechanism:* separately from scale,
  that paper obtains contamination by **fine-tuning on a contamination set repeated a set number of times
  inside the fine-tuning data**, across three regimes — LoRA r=8, LoRA r=256 and full fine-tuning — and the
  regime in which CDD works there is full fine-tuning, which updates every parameter rather than an adapter
  (§2.4). The contamination this design studies **may instead arise during pretraining or post-training of
  the Instruct checkpoints**, where benchmark text is not injected on a controlled repetition schedule and
  the exposure duration is not separately set. That paper's own Limitations draws the same distinction:
  *"pre-training contamination, where benchmark data appears in the original training corpus without
  explicit repetition, may produce different dynamics."* The threshold governing CDD is stated there as an
  interaction of model size, adapter rank and training duration, and its working regime imposes no rank
  restriction at all, so what separates that setting from ours is not a count of trainable parameters but
  repetition-injected fine-tuning versus unrepeated pretraining or post-training exposure. We therefore
  treat CDD's operability at this scale as an open main-study
  question, not an assumption or an engineering-validation criterion (§4.6–§4.7). Its temporal-proxy AUC
  cannot by itself separate detector-floor behavior from temporal-label error, much less attribute either
  pattern to scale or to the form in which memorization arose.
- **Olmo3's corpus-reference statuses are model-local and method-conditional.** The corpus search in §5,
  step 5 supplies direct positive matches and searched non-matches for Olmo3 only; non-detection is not
  proof of absence. Qwen2.5's and Llama-3.1's training corpora remain closed, so those arms keep the proxy
  labels and their unidentified error rate. Olmo3 supplies positive-match evidence about how a
  declaration-based temporal split relates to operational corpus-search results on this benchmark family
  — the boundary for these two arms comes from the model cards' declared date, not from a release date
  (§4.2) — but it does not identify *e* and must not be presented as ground truth for the other arms.
- **Scale mismatch in the BNB-nf4 effect-size prior.** The 32% BNB-nf4 drop cited in §2.7 was
  measured in arXiv:2505.20276 on **Llama-3.1-70B**. Our Llama arm, Llama-3.1-8B, shares the family and
  data recipe but is roughly 9× smaller, and quantization fragility is known to vary with scale. The
  prior provides only a scale- and task-mismatched stress-case motivation. Neither this figure nor
  calibration-free status establishes the largest detector effect. Estimate the actual effect in the
  frozen main study without data-dependent resizing.
- **Cutoff uncertainty on the Llama-3.1-8B arm.** The detected knowledge boundary is a behavioral
  diagnostic, not a verified training-data cutoff. The 2023-04-01 sensitivity boundary yields no LCB
  possible-exposure items, so Llama's own sensitivity effect is not estimable (§4.2). Misclassified
  temporal items need not cause benign attenuation: date-dependent difficulty or topic changes can
  change an association's magnitude or sign. Do not claim robustness from the unavailable contrast.
- **Base-rate and difficulty confounding** (§4.5.3) remains a threat despite the chosen log-odds scale.
  Required stratification diagnostics can reveal poor overlap or effect heterogeneity, but cannot prove
  their absence. Additional matched analyses are exploratory and do not replace the fixed contrast.
- **The two proxy groups may differ in item characteristics, not only in possible exposure.** Items
  published before and after 2025-01-01 can differ in difficulty, source platform and problem-statement
  length; we have not measured these distributions, and the possibility is stated here as a conjecture
  rather than an observed property of `release_v6`. Any such difference can move a detector's proxy AUC
  away from 0.5 with no exposure involved, and if quantization changes how sensitive a detector is to
  difficulty or length, a C4 ranking reversal can follow from that alone. Q1b's estimand is the temporal
  proxy AUC (§4.5.2), so this does not redefine what is estimated; it blocks reading a proxy-AUC change as
  a change in exposure signal. §4.5.3's stratification diagnostics address Q2 only, so we additionally
  report descriptive statistics for difficulty, source platform and item length by proxy group. Those
  statistics are descriptive: they do not change the labels, the estimand, or the fixed analysis family.
- **Inference-implementation numerics are not fully separated from the quantization effect.** All four
  conditions run on one inference stack (§4.3), so no precision contrast is simultaneously a change of
  inference engine, and C1–C3's bf16→BNB-nf4 contrast stays inside the bitsandbytes load-time path. What
  remains is the AWQ-versus-BNB comparison: the two paths use different quantization kernels, so a
  difference measured there contains the kernel and implementation difference alongside the weight-format
  difference, and this design cannot separate the two. This is the same class of confound the paper gives
  for excluding QAT-shipped checkpoints, whose format would require a second inference stack (§4.1). The
  AWQ contrasts are exploratory (§4.5.6) and their differences are not attributed to the weight format
  alone.
- **AWQ calibration exposure.** The AWQ-int4 condition is the only rung that sees a calibration corpus
  (§4.3); the other three never do. If that corpus overlaps evaluation prompts or reference solutions, the
  AWQ arm's detector scores and pass@1 carry that exposure as well as the weight format. §4.3 requires the
  overlap check and a single frozen calibration artifact per model before the main run, but that check is
  not proof of semantic non-overlap.

- **Temporal exposure labels carry unidentified error** (§4.5.2). `no-match-found` is not verified
  non-exposure even on Olmo3, so no arm yields a binary error-rate estimate. The symmetric attenuation
  table is a construct-validity sensitivity model only, not a correction or an empirical sizing input.
- **TRACER is a reimplementation, not the authors' code.** No public code release is confirmed for
  TRACER; §5, step 5 rebuilds it from the paper's published prompts, embedding model, and thresholds.
  Reimplementation fidelity is therefore itself a threat — assessed, but only on the Olmo3 arm, by
  descriptively comparing the reimplementation with confirmed-positive corpus-reference evidence (§5, step 5). Its accuracy on the
  closed-corpus arms' conditions remains unknown; TRACER results are descriptive support and do not
  redefine Q1b's temporal-proxy estimand.
- **Scope limits.** Five models (7B–32.5B), four quantization configurations, code generation only,
  predominantly Python. Findings need not generalize to other architectures (e.g., MoE), other
  tokenizers, or other domains (e.g., natural-language QA), and we do not claim they do.

---

## 7. Limitations

- No causal claim about **exposure** is possible without random assignment of contamination, which cannot
  be done on off-the-shelf pretrained models (§6). This limit is specific to the exposure axis. The
  precision axis is a controlled manipulation of one frozen checkpoint, so C1–C3, which use no exposure
  label at all, are limited by inference-implementation numerics (§6) rather than by the absence of random
  assignment; Q1b and Q2 bring the exposure proxy in and are reported as associations.
- Q1 declares no equivalence margin (§3.2), so a nonsignificant C1–C4 result is inconclusive: it is
  reported as such and never as evidence that detector scores are stable under quantization. C4's power at
  the achieved group sizes is not guaranteed either (§4.5.6).
- Q2 may remain underpowered even after all mitigations in §4.5; if so, it is reported as an
  interval-bounded secondary result, not as a significance claim, and the paper's contribution
  claim does not depend on it clearing significance.
- CDD may have insufficient operational proxy AUC to separate the temporal proxy groups (§4.7). It remains
  in the fixed analysis family; chance-level main-study performance is reported with uncertainty and is not
  promoted to evidence that CDD is inoperative against verified contamination.
- Cutoff evidence quality differs across arms (§4.2): Olmo3's boundary rests on official model-card declarations,
  Llama-3.1-8B's on a declaration with an independent knowledge-boundary diagnostic, and Qwen2.5's — absent an unambiguous
  declaration — on the release-date upper bound. The pre/post split's precision is therefore
  arm-dependent; boundary evidence is tabulated per arm, each arm's `possible-exposure` label uses that
  arm's own bound, and only the shared control is restricted to dates after every primary bound (§4.2).
- The scope is limited to five code-generation-capable dense transformer models (7B–32.5B) and four
  quantization configurations; no model above 32.5B is tested — a hard single-device compute constraint
  (§4.1) — and generalization beyond this scope, upward in scale included, is not claimed.

---

## References

Identified by arXiv ID, grouped by topic. Sources without an arXiv ID are listed under *Other references*
at the end.

**Contamination — surveys**
- arXiv:2404.00699 — *A Comprehensive Survey of Contamination Detection Methods in Large Language Models* (TMLR 2025)
- arXiv:2502.14425 — Cheng, Chang, Wu (2025) — *A Survey on Data Contamination for Large Language Models*
- arXiv:2502.17521 — *Recent Advances in Large Langauge Model Benchmarks against Data Contamination: From Static to Dynamic Evaluation* [sic, original typo preserved]
- arXiv:2605.26133 — Tong, Sun, Nguyen (2026) — *Pretraining Data Exposure in Large Language Models: A Survey of Membership Inference, Data Contamination, and Security Implications*

**Detection methods used or named in this design**
- arXiv:2402.15938 — Dong, Jiang, Liu, Jin, Gu, Yang, Li — *Generalization or Memorization: Data Contamination and Trustworthy Evaluation for Large Language Models* (ACL 2024; introduces CDD, Contamination Detection via output Distribution)
- arXiv:2310.16789 — Shi, Ajith, Xia, Huang, Liu, Blevins, Chen, Zettlemoyer — *Detecting Pretraining Data from Large Language Models* (introduces Min-K% Prob)
- arXiv:2309.10677 — Li — *Estimating Contamination via Perplexity: Quantifying Memorisation in Language Model Evaluation*
- arXiv:2401.17377 — Liu et al. — *Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens* (candidate index implementation, §5 step 5)

**Temporal-split contamination measurement**
- arXiv:2310.10628 — *Data Contamination Through the Lens of Time*
- arXiv:2403.07974 — Jain, Han, Gu, Li, Yan, Zhang, Wang, Solar-Lezama, Sen, Stoica (2024) — *LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code*
- arXiv:2511.12116 — *LLMLagBench: Identifying Temporal Training Boundaries in Large Language Models*
- arXiv:2504.14655 — Xia, Shen, Wang, Liu, Sun, Wu, Hu, Xu (2025) — *LeetCodeDataset: A Temporal Dataset for Robust Evaluation and Efficient Training of Code LLMs*

**Effect sizes of contamination**
- arXiv:2501.18771 — Kocyigit, Briakou, Deutsch, Luo, Cherry, Freitag (2025) — *Overestimation in LLM Evaluation: A Controlled Large-Scale Study on Data Contamination's Impact on Machine Translation*
- arXiv:2403.04811 — *Quantifying Contamination in Evaluating Code Generation Capabilities of Language Models* (ACL 2024)
- arXiv:2506.02791 — Yang, Lin, He, Wang, Sun, Liu, Xu, Wang, Yu, Liang (2025) — *Contamination Means Overestimation? A Fine-Grained Empirical Study in Code Intelligence*
- arXiv:2507.19219 — Liang, Yu, Zhang, Ye, Hu (2025) — *How Much Do Large Language Model Cheat on Evaluation? Benchmarking Overestimation under the One-Time-Pad-Based Framework*

**Limits of contamination detection**
- arXiv:2311.04850 — *Rethinking Benchmark and Contamination for Language Models with Rephrased Samples*
- arXiv:2602.12413 — *Soft Contamination Means Benchmarks Test Shallow Generalization*
- arXiv:2402.02823 — Dekoninck, Müller, Baader, Fischer, Vechev (2024) — *Evading Data Contamination Detection for Language Models is (too) Easy*
- arXiv:2409.09927 — *Towards Data Contamination Detection for Modern Large Language Models: Limitations, Inconsistencies, and Oracle Challenges*
- arXiv:2603.03203 — *No Memorization, No Detection: Output Distribution-Based Contamination Detection in Small Language Models*

**Code-benchmark contamination**
- arXiv:2605.24079 — *TRACER: A Semantic-Aware Framework for Fine-Grained Contamination Detection in Code LLMs*
- arXiv:2512.13961 — *Olmo 3* (Ai2 technical report; cited for its per-stage training-data decontamination methodology, §5 step 5)
- arXiv:2411.10842 — Cao, Chen, Zhang, Lo, Cheung (2024) — *CODECLEANER: Elevating Standards with A Robust Data Contamination Mitigation Toolkit*
- arXiv:2503.06643 — Guan, Wu, Yuan, Li (2025) — *Is Your Benchmark Still Useful? Dynamic Benchmarking for Code Language Models*
- arXiv:2503.13572 — Wang, Shao, Bhandari, Mankali, Karri, Sinanoglu, Shafique, Knechtel (2025) — *VeriContaminated: Assessing LLM-Driven Verilog Coding for Data Contamination*

**Benchmarks and model suites named in the text**
- arXiv:2107.03374 — Chen et al. — *Evaluating Large Language Models Trained on Code* (HumanEval)
- arXiv:2108.07732 — Austin et al. — *Program Synthesis with Large Language Models* (MBPP)
- arXiv:2305.01210 — Liu et al. — *Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation* (EvalPlus)
- arXiv:2304.01373 — Biderman et al. — *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling*

**Post-hoc decontamination**
- arXiv:2509.15218 — Hou, Jiao, Hu, Li, Lam, Zhang, Lu (2025) — *LNE-Blocking: An Efficient Framework for Contamination Mitigation Evaluation on Large Language Models*
- arXiv:2601.19334 — Chai, Zhe, Sakuma (2026) — *When Benchmarks Leak: Inference-Time Decontamination for LLMs*
- arXiv:2506.04142 — Zhu, Tu, Jin, Hou, Li, Zhao (2025) — *Establishing Trustworthy LLM Evaluation via Shortcut Neuron Analysis*
- arXiv:2605.21543 — Liu, Zeng, Wei (2026) — *Provable Joint Decontamination for Benchmarking Multiple Large Language Models*

**Quantization**
- arXiv:2503.07103 — *Evaluating the Impact of Post-Training Quantization on Large Language Models for Code Generation*
- arXiv:2507.09665 — *Is Quantization a Deal-breaker? Empirical Insights from Large Code Models*
- arXiv:2506.22776 — *Smaller = Weaker? Benchmarking Robustness of Quantized LLMs in Code Generation*
- arXiv:2505.20276 — *Does quantization affect models' performance on long-context tasks?*
- arXiv:2208.07339 — Dettmers et al. — *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale* (the BNB int8 condition, §4.3)
- arXiv:2305.14314 — Dettmers et al. — *QLoRA: Efficient Finetuning of Quantized LLMs* (introduces the NF4 data type, §4.3)
- arXiv:2306.00978 — Lin et al. — *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* (the AWQ-int4 condition, §4.3)

**Quantization × unlearning/memorization**
- arXiv:2410.16454 — *Catastrophic Failure of LLM Unlearning via Quantization* (ICLR 2025)
- arXiv:2605.15138 — *Forgetting That Sticks: Quantization-Permanent Unlearning via Circuit Attribution*
- arXiv:2508.00128 — *How Quantization Impacts Privacy Risk on LLMs for Code?*
- arXiv:2607.25451 — *Bits and Memories: Measuring Verbatim Extraction Across LLM Quantization*

**Other references**
- Hanley, J. A., & McNeil, B. J. (1982). The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology*, 143(1), 29–36. (AUC standard-error formula, §4.5.2)
- DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics*, 44(3), 837–845. (joint AUC covariance for C4, §4.5.6)
