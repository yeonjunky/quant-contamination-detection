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

# Illustrative bf16 base rates used in that same worked table (paper §4.5.3,
# "these two figures are illustrative values for a Qwen-class instruction-tuned
# model" — NOT a claim about every model. Main-study base rates are reported
# from the frozen analysis and never used to redesign the study.
BASE_RATE_HUMANEVAL_ILLUSTRATIVE = 0.85
BASE_RATE_LCB_POST_ILLUSTRATIVE = 0.35

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
# Version the scoring rule separately from generation settings: old validation
# scores used alpha * l_max even when every output was shorter than l_max.
CDD_SCORE_DEFINITION = "actual-max-truncated-length-v2"
# Original CDD paper's fixed, 7B-calibrated decision threshold. arXiv:2603.03203
# re-selects xi per condition via Youden-index maximization on its own eval
# set and explicitly flags this as an optimistic oracle ("gives CDD every
# advantage") — see detectors/threshold.py. AUC (threshold-independent) is
# Q1b's primary metric; this constant is only relevant for descriptive
# point-accuracy reporting.
CDD_XI_FIXED = 0.01

# --- Frozen decoding settings (paper §4.4, "Frozen scoring protocol") -------
# §4.4: "Every decoding setting is specified explicitly and the checkpoint's
# own `generation_config` is not followed: `top_p=1.0`, top-k sampling
# disabled, `repetition_penalty=1.0`, and no length penalty or minimum-length
# constraint. The same settings apply to the greedy reference output, which
# differs from the samples only in that sampling is off."
#
# These were an engineering default in models/loader.py until §4.4 pinned
# them; they now live here because the paper fixes them, and because
# real_run.py has to record the same numbers in the run manifest.
GENERATION_MAX_NEW_TOKENS = 512  # §4.4's "512-token generation cap"
DECODING_TOP_P = 1.0
# transformers disables top-k with `top_k=0`, not `None` (a `None` field is
# treated as "unset" and is refilled from the checkpoint's generation_config;
# transformers 5.14.1 generation/utils.py:1279-1282 builds the top-k warper
# only when `top_k is not None and top_k != 0`, and :1769 is the line that
# refills unset fields from the checkpoint).
DECODING_TOP_K_DISABLED = 0
DECODING_REPETITION_PENALTY = 1.0
DECODING_LENGTH_PENALTY = 1.0
DECODING_MIN_NEW_TOKENS = 0
# The greedy reference output uses temperature 1.0 with sampling off, so the
# recorded setting is a true no-op rather than a value transformers would
# ignore.
DECODING_GREEDY_TEMPERATURE = 1.0
FOLLOW_CHECKPOINT_GENERATION_CONFIG = False

# --- Dataset conditions (CLAUDE.md §4.2 / paper §4.2) -----------------------

HUMANEVAL_N_ITEMS = 164  # hard ceiling, evalplus-pinned
MBPPPLUS_N_ITEMS = 378

# The fixed common boundary (paper §4.2/§5 step 3): "On or after 2025-01-01,
# the first day after the latest model-level cutoff" — the two Olmo Instruct
# cards' `Date cutoff: Dec. 2024` makes 2025-01-01 their first post-boundary
# date, and it is the latest of the five arms' bounds. This is a fixed design
# value, not an open question; scripts/run_main.py defaults to it and keeps a
# CLI override only for the §4.2 boundary-sensitivity re-runs.
LCB_SHARED_CONTROL_BOUNDARY = "2025-01-01"

# Paper §5 step 3's planning target for the primary LCB condition. Under
# `release_v6` at the common boundary the availability envelope is pre 873 /
# shared control 182 / total 1,055, so this target is **unmet** and Q2 stays a
# secondary, interval-focused analysis. Kept as the stated target the design
# was sized against, not as an expected count.
LCB_TARGET_N_PER_CONDITION = 1000
