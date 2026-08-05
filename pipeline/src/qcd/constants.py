"""Single source for statistical/design constants named across CLAUDE.md and
paper/paper_draft.md. Never re-type these numbers elsewhere (CLAUDE.md §3.3
discipline: mixing values computed under different assumptions in one table
is the recurring failure mode this repo has hit before).

Every constant below is either (a) a design choice fixed by the paper draft,
or (b) a value independently re-derived from the paper's worked tables in
paper/paper_draft.md and cross-checked against CLAUDE.md §4.1 during pipeline
construction (see analysis/auc.py, analysis/logodds.py for the derivations).
Do not "fix" these to a different value without re-checking the draft first.
"""

from __future__ import annotations

# --- Statistical design (CLAUDE.md §4.1) -----------------------------------

ALPHA = 0.05  # two-sided significance level used throughout
POWER_TARGET = 0.80

# Item-conditional difficulty random-effect SD used in the §4.5.3 base-rate
# confound worked table (HumanEval/LCB-post spurious-interaction table).
# Re-derived by numerical calibration against paper_draft.md's β=0.50 row
# (5.6pp/7.7pp/−2.2pp) during pipeline construction — see analysis/logodds.py.
DIFFICULTY_SIGMA = 1.5

# Cross-precision item-level correlation implied by DIFFICULTY_SIGMA at p=0.5,
# fixed by high-precision numerical integration (NOT Monte Carlo — MC runs have
# historically produced 0.291–0.293 and caused a 549-vs-556 table inconsistency;
# see revision_provenance.md (e)). Paper §4.5.3 decomposition row 3:
# n = 785 × (1 − r) = 555.
IMPLIED_R_AT_P50 = 0.293089

# §4.5.6 confirmatory family: 4 pre-specified tests, Holm-corrected. Worst-case
# multiplier at alpha/4 (z_{1-0.05/8} + z_{0.80}); Q1a needs ≈124 items at
# d=0.3, ≈279 at d=0.2 under this sizing.
CONFIRMATORY_FAMILY_SIZE = 4
HOLM_WORST_CASE_MULTIPLIER = 3.3393

# Illustrative fp16 base rates used in that same worked table (paper §4.5.3,
# "these two figures are illustrative values for a Qwen-class instruction-tuned
# model" — NOT a claim about every model; real base rates are measured per
# model in the pilot, §4.7 item (d)).
BASE_RATE_HUMANEVAL_ILLUSTRATIVE = 0.85
BASE_RATE_LCB_POST_ILLUSTRATIVE = 0.35

# CDD pilot gate (§4.6): break-even baseline AUC below which CDD is dropped
# from the primary analysis. The draft states "≈0.79"; 0.7936 is the exact
# value solving detection_limit(n=542, AUC, r=0.8) == quantization_delta_auc(AUC)
# (CLAUDE.md §4.1 records this same value). Use this constant, not 0.79 — the
# rounded figure is for prose, not for a threshold comparison in code.
CDD_GATE_AUC = 0.7936

# §4.6's assumed quantization-induced reduction in CDD separation, expressed
# on the binormal d'-scale (d' = sqrt(2) * Phi^-1(AUC)); reproduces the
# table's ΔAUC column exactly (0.002/0.010/0.019/0.026/0.019 for
# AUC=0.52/0.60/0.70/0.85/0.95).
CDD_GATE_ASSUMED_SEPARATION_REDUCTION = 0.10

# §4.6's reference item count and cross-precision correlation for the gate
# table (542-item, r=0.8 reference point).
CDD_GATE_REFERENCE_N = 542
CDD_GATE_REFERENCE_R = 0.8

# §4.5.2's baseline AUC assumption for the Q1b SE(AUC)/label-noise tables.
Q1B_REFERENCE_AUC = 0.70

# --- CDD sampling protocol (Dong et al. 2024, as replicated and re-stated
# verbatim in arXiv:2603.03203 — pipeline/pdfs/2603.03203.pdf, "Sampling" /
# "Edit distance computation" / "Peakedness" / "Classification" subsections).
# Do not change these without re-reading that source; CLAUDE.md §3.1 —
# formulas must be verified against the PDF, not guessed. -------------------

CDD_N_SAMPLES = 50  # "We use n=50, matching the original paper."
CDD_SAMPLE_TEMPERATURE = 0.8
CDD_GREEDY_TEMPERATURE = 0.0
# Similarity threshold α in Peak(M;x) = (1/n) * sum_i I(ED(s_i, s_greedy) <= α*l).
# NOTE: this is the *edit-distance* alpha from Dong et al., unrelated to the
# statistical significance ALPHA=0.05 above — same numeric value, different
# quantity. Keep them as separate constants so a future edit to one doesn't
# silently change the other.
CDD_EDIT_DISTANCE_ALPHA = 0.05
CDD_MAX_TOKENS = 100  # l_max: sequences truncated to this length before ED.
# Original CDD paper's fixed, 7B-calibrated decision threshold. arXiv:2603.03203
# re-selects xi per condition via Youden-index maximization on its own eval
# set and explicitly flags this as an optimistic oracle ("gives CDD every
# advantage") — see detectors/threshold.py. AUC (threshold-independent) is
# Q1b's primary metric; this constant is only relevant for descriptive
# point-accuracy reporting.
CDD_XI_FIXED = 0.01

# --- Dataset conditions (CLAUDE.md §4.2 / paper §4.2) -----------------------

HUMANEVAL_N_ITEMS = 164  # hard ceiling, evalplus-pinned
MBPPPLUS_N_ITEMS = 378
LCB_TARGET_N_PER_CONDITION = 1000  # target, not yet confirmed (paper §5 step 3)
