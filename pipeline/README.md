# pipeline/

Data-collection pipeline for the quantization contamination-detection paper.
See `../pipeline_build_plan.md` at the repo root for the full design and
rationale; this file is just setup + run instructions.

Olmo training-data string-match corpus-reference work is documented separately in
[`OLMO_GROUND_TRUTH.md`](OLMO_GROUND_TRUTH.md). It is CPU-only and does not turn
partial streaming scans into clean labels.

## Manuscript synchronization — 2026-09-09

CDD now uses `alpha * l`, where `l` is the longest actual output after the 100-token cap,
including greedy and all samples. It no longer uses `alpha * 100` for short outputs.
The main-run manifest records `actual-max-truncated-length-v2`, alpha and the token cap;
resuming an older score directory fails the configuration check. Older smoke outputs remain
historical engineering evidence only and must not be mixed with study scores.

The canonical paper §4.5.6 now fixes C1–C4 to Qwen2.5-32B-Instruct. C1–C3 use all
1,055 LCB items; C4 uses 690/182 proxy groups and literal reversal of the mean probability
AUC versus CDD.

**Updated 2026-09-18 — the analysis layer is now wired.** `analysis/confirmatory.py`
implements the whole family: the three paired t-tests, the C4 reversal test with its DeLong
covariance, Holm over the four slots, and the "not estimable" rule that keeps a slot at p=1
instead of dropping it. `scripts/run_analysis.py` is the entry point, and the model, the
contrast, the detector-to-slot mapping and the item sets are module constants there, not CLI
flags. Q2's reported β_QE interval comes from the item-stratified conditional logistic Wald
fit in `analysis/conditional_logit.py`, run separately per model, one quantized level against
bf16 at a time; `mixed_effects.py`'s variational-Bayes fit still returns an approximate
posterior SD, which §4.5.5 forbids reporting as an interval, so it supplies point estimates
and variance components only. `scripts/verify_interval_coverage.py` runs §4.5.5's synthetic
coverage check that fixes which method supplies the reported interval, and `run_analysis.py`
refuses to run without its record. AWQ calibration is no longer selected after the fact —
see "AWQ calibration" below.

**Also 2026-09-18 — decoding settings pinned, raw schema widened.** Paper §4.4's decoding
settings are now fixed in `constants.py` and handed to `generate()` as a `GenerationConfig`
object (`top_p=1.0`, top-k disabled with `top_k=0`, `repetition_penalty=1.0`, no length
penalty or minimum-length constraint, 512-token cap), and the adapter replaces each
checkpoint's own `generation_config` with one carrying nothing but its stop/pad token ids, so
the checkpoint contributes nothing to decoding. The stored per-token log-probabilities are the
**raw** values read from `outputs.logits`, before the logits processors run. The raw tables
gained the columns §4.4 asks to be recorded alongside a probability-detector score:
`truncated_at_cap` and `max_new_tokens`, a `decoding_settings_id` pointing at the manifest's
full resolved record, and — on the greedy row — the scored text's sha256, character span,
token indices and `chat_template_id`. A new `chat_templates.parquet` holds the rendered
template text, one row per model/precision, so it is stored once per run instead of on every
generations row. The common boundary 2025-01-01 is a constant
(`constants.LCB_SHARED_CONTROL_BOUNDARY`), not a per-script default.

## What's built vs. what's deferred

The following GPU-free components are built and tested: dataset
loaders (LiveCodeBench/HumanEval+/MBPP+), generation sampling + cache,
sandboxed scoring (real subprocess execution, not mocked), all three
detectors (CDD/perplexity/Min-k% Prob), statistical analysis helpers
(power tables, log-odds base-rate model, variational-Bayes mixed-effects GLMM), the frozen
analysis layer (§4.5.6's C1–C4 family in `analysis/confirmatory.py`, §4.5.5's per-model β_QE
interval in `analysis/conditional_logit.py`, the synthetic interval-coverage check in
`analysis/coverage.py`, raw-tree loading and the §4.4 truncation rates in
`analysis/study_inputs.py`, driven by `scripts/run_analysis.py`), development-only
synthetic diagnostics, model–item temporal-label materialization, raw-data/manifest writers,
and the mock dry run
(`scripts/run_dry_run.py`) exercising all of it end-to-end.

**Built and validated on real H100 hardware (2026-08-15):**
`models/loader.py`'s real `generate()`/`score_logprobs()` for the bf16 and
bitsandbytes (int8/nf4) backends, exercised end-to-end by
`scripts/run_smoke_test.py` against Qwen2.5-7B-Instruct/BNB-nf4 — real
quantized load (peak 6.69GB), real sampling, real sandboxed code execution,
real detector scoring, real raw-data writer, all checklist items passing.
`run_main.py` drives the full four-level ladder
(`Quant.BF16`/`BNB_INT8`/`BNB_NF4`/`GPTQ_AWQ_INT4`); the AWQ rung additionally
needs its offline checkpoint built first (see "AWQ calibration" below). The
former `run_pilot.py` entrypoint is
retired because engineering validation rows must not be aggregated as study estimates.

**Real finding from that run, fixed same session (2026-08-15):**
`real_run.py`'s `_assemble_candidate_code()` used to assume HumanEval+/MBPP+
candidate code was `item.prompt + completion_text` verbatim, but the roster
is entirely -Instruct models queried through a chat template — the model
answers with prose plus a fenced code block, not a raw continuation, so the
assembled candidate for these two datasets was not runnable Python at all —
the sandbox was scoring the prose, not the solution. Fixed by stripping the
markdown fence first (`_strip_markdown_fence`) and reusing evalplus's own
post-processing (`evalplus.sanitize.sanitize`/`code_extract` — the same step
evalplus's leaderboard runs on LLM output) rather than reimplementing
extraction; see `_assemble_candidate_code`'s docstring for the
HumanEval+/MBPP+-vs-LCB extractor choice. The LCB path was subsequently
validated on real Qwen2.5-7B BNB-NF4 outputs spanning pre/post and
stdin/functional tasks; see `scripts/run_lcb_smoke_test.py`. What the
validation run established is that extraction produces runnable candidate
code; per §4.6 no pass rate from it is recorded here or anywhere else.

**Built and validated on real H100 hardware, same day:** the
`Quant.GPTQ_AWQ_INT4` quant rung (`models/loader.py`'s `_load_gptq_or_awq`)
— via **AWQ only; GPTQ itself was not implemented.** Resolved "open
assumption #1" differently than originally planned: llm-compressor (AWQ)
uniformly for all five roster models, not GPTQModel, not a per-model
GPTQ/AWQ split — GPTQModel's own architecture registry has no `olmo3` entry
(checked against source), while llm-compressor has no per-architecture
registry and was confirmed to work on Olmo3-7B-Instruct with no workaround
needed. Full rationale — including a "worth trying GPTQModel later" note —
lives in `pipeline_build_plan.md`'s "Open assumptions" #1, not in code. See
`scripts/quantize_model.py` for the offline quantization step (quantize
once and save, unlike bnb's load-time quantization).

**AWQ calibration — fixed, not selected.** Paper §4.3 freezes exactly one
calibration artifact per model, so `scripts/quantize_model.py` takes only a
model name (plus `--lcb-release` for the overlap check) and quantizes every
model with the same **code** calibration set: `flytech/python-codes-25k` at a
pinned revision, 256 rows shuffled with seed 42, max sequence length 512, read
from the dataset's own `text` column with no chat template applied. There is
no `--calibration` argument, no code-vs-chat comparison run, and no copying a
winner into place afterwards — the script writes straight to
`data/quantized/<model>-awq/`, the one canonical path
`models/loader.py::_quantized_checkpoint_dir` reads:

```bash
python scripts/quantize_model.py Qwen2.5-7B-Instruct
```

Next to the checkpoint it writes the two records §4.3 requires:
`quantization_manifest.json` (dataset revision, the seed, the sha256 of every
selected calibration row and of the selected-row list as a whole, the
tokenizer, the recipe, the resolved AWQ group size, installed versions) and
`calibration_overlap_report.json` (a 13-token n-gram lexical check of the
calibration text against LiveCodeBench `question_content`, the HumanEval+ and
MBPP+ prompts, and those two arms' canonical solutions). The overlap check is
a *pre-execution requirement*, so `_load_gptq_or_awq` refuses to load a
checkpoint that has no report at all — a nonzero overlap count is a finding to
adjudicate and record, not an automatic block. The earlier code-vs-chat
calibration comparison was engineering work; its outputs are not evidence, the
chat set does not enter the main run, and this script does not produce it.

**Real finding along the way:** `AutoModelForCausalLM.from_pretrained()` on
our AWQ (W4A16, asymmetric) checkpoints uses ~15-16GB peak GPU memory for a
7B model, not the ~4-5GB the on-disk int4 size would suggest — bnb-nf4's
dedicated kernels keep weights packed through inference; plain-transformers
AWQ decompression apparently doesn't, matching a known compressed-tensors/
transformers rough edge with asymmetric zero-points
(vllm-project/llm-compressor#1550). Still fits comfortably on the 80GB H100
for the 7B/8B arms; worth watching for the 32B arms later, where it would
erode the memory headroom the paper's own compute table assumed AWQ/GPTQ
would provide over bf16.

**Still not empirically exercised:** Llama-3.1-8B-Instruct and the two 32B
models (Qwen2.5-32B, Olmo3.1-32B) — deliberately deferred to the frozen main
run, not attempted in this validation pass.

## Environments

Machines and install profiles this design targets:

- **A CUDA laptop (e.g. RTX 4060, 8GB VRAM)** — mock-only profile for the
  dry run, optionally layered with the real-smoke profile for the nf4 smoke
  test.
- **H100 box (validated 2026-08-15)** — `requirements-h100.txt` is now a
  pinned lockfile covering bf16/bnb *and* GPTQ/AWQ (llm-compressor):
  torch 2.13.0+cu130, transformers 5.14.1, bitsandbytes 0.50.1,
  accelerate 1.14.0, llmcompressor 0.13.0, compressed-tensors 0.18.0,
  statsmodels 0.14.6, scipy 1.18.0, ...
  Note torch/transformers/numpy are newer here than what was first installed
  — `pip install llmcompressor` silently pulled a newer torch as a
  transitive dependency partway through this session; the bnb-nf4 smoke
  test was re-verified afterward and still passes (see requirements-h100.txt's
  own comment for the full sequence). GPTQModel is deliberately not
  installed — see "What's built" above for why.
- **A Mac (Apple Silicon, no CUDA)** — mock-only profile plus the real,
  GPU-free pieces (dataset downloads, sandboxed code execution, statistics):
  everything under "Running the dry run" and the full `pytest` suite below.

```bash
# from pipeline/
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-local.txt        # mock-only, no GPU library at all;
                                             # includes the CPU-only statistics stack
                                             # (statsmodels, scipy) the analysis layer imports
pip install -e .

# only when running against a real GPU backend (smoke test or H100 main run):
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-smoke.txt        # local laptop-scale smoke test
# or: pip install -r requirements-h100.txt   # full pinned H100 stack
```

## Local smoke-test checklist

**Run and passing (2026-08-15, real H100), against bf16/bnb *and* AWQ.**
`scripts/run_smoke_test.py` runs this as an automated checklist against 5
real HumanEval items and exits non-zero if any check fails —
`--model` and `--quant` select the arm (every model in `models/registry.py`,
all four precisions of the §4.3 ladder), and `--checkpoint-path` bypasses
`load_model()`'s canonical-path resolution to load a checkpoint sitting
somewhere else on disk. Output goes to the validation-only namespace
`data/raw/validation/smoke_test/<quant>/`, with a manifest recording
`study_phase="engineering_validation"`.

Per paper §4.6 the checklist tests *properties* of the numbers, not their
values: pass rates and detector scores are computed so the range checks have
something to check, and are then neither printed, written to the parquet rows,
nor compared across items or precisions. Memory and wall-clock below are from
the default bnb-nf4 run; the AWQ runs passed the same checklist with peak
memory in the ~15-16GB band documented above instead:

- [x] quantized load fits in a plausible memory band (per model *and* precision — see `plausible_peak_gb()` and the AWQ memory finding above) — bnb-nf4 peak 6.69GB, no OOM on the 80GB card
- [x] teacher-forced logprob scoring returns finite values (no NaN/-inf), both for the completion and for the fixed-prompt `score_prompt_logprobs()` pass
- [x] repeated T=0.8 samples for the same item actually differ
- [x] the real sandboxed code-execution path runs, and its partial pass rate lands in [0, 1]
- [x] CDD / perplexity / Min-k% detectors run end-to-end without dtype/shape errors, and their scores are finite and in range
- [x] the real raw-data writer's output matches the mock's schema exactly
- [x] the validation manifest is written, carrying §4.3's resolved library defaults read off the loaded model
- [x] per-item wall-clock is sane — model load 10.7s (weights cached), ~10-19s/item
- [x] `pip freeze` saved to `envs/local-smoke-freeze.txt`

Each smoke-test invocation uses a fresh, ephemeral generation cache. Its
completion-confidence cross-check still calls history-dependent
`score_logprobs()`, so a cross-process cache hit would skip the `generate()`
call that records its prompt. The real Q1 probability-detector path instead
uses independent `score_prompt_logprobs(item_id, prompt)` and is safe on
persistent generation-cache hits.

## Running the dry run

```bash
python scripts/run_dry_run.py
```

Exercises the implementation path on ~10-20 synthetic items with zero
GPU/downloads. Its diagnostics are validation-only and are not manuscript evidence,
power inputs, or detector-selection inputs. Also runnable as
`pytest tests/test_mock_pipeline_end_to_end.py`.

## Engineering validation and the main run

```bash
python scripts/run_dry_run.py
python scripts/run_smoke_test.py --help
python scripts/run_lcb_smoke_test.py --help
```

Both smoke tests take `--model` and `--quant`, so any roster model at any of the
four precisions can be validated before the main run touches it. They write to
the validation-only namespace — `data/raw/validation/smoke_test/<quant>/` and
`data/raw/validation/lcb_smoke_test/` (repo-root-anchored, inside the gitignored
`/data/`) — and each writes a manifest recording
`study_phase="engineering_validation"`. The dry run writes to a temporary
directory.

The separation is enforced in code, not by directory naming: `study_phase` is a
required field of every manifest (`io/manifest.py`), and
`io/manifest.py::require_main_study` is the first thing the analysis entry point
calls, so a validation tree is refused before a single parquet file is opened.
Inspect only implementation properties such as loading, schema integrity, finite
log-probabilities, sandbox behavior, memory, and throughput. Do not calculate
effect sizes, AUCs, pass rates, correlations, power, or detector rankings from
validation rows.

```bash
python scripts/run_main.py --help
```

`run_main.py` is the only study-data driver. It runs all five
`MAIN_ANALYSIS_MODELS` across the full four-level ladder and writes to
`data/raw/main` (repo-root-anchored) with `study_phase="main_study"`. Freeze the
operational configuration before starting it. `run_pilot.py` and
`aggregate_pilot.py` are deprecated compatibility entrypoints that intentionally
exit without running or aggregating data.

## Analysis

```bash
python scripts/verify_interval_coverage.py --replications 200   # once, before the main run
python scripts/run_analysis.py ../data/raw/main
```

`verify_interval_coverage.py` is §4.5.5's synthetic-data coverage check. It is
CPU-only, needs no GPU, no network and no run directory, and touches no study
data; it writes `analysis_artifacts/interval_coverage_check.json` — a tracked
path, not a temp directory, because the analysis manifest has to carry its
generating parameters, replication count and achieved coverage. Its result is
what fixes which method supplies the reported β_QE interval; it is not a study
result.

`run_analysis.py` is the frozen analysis driver. It gates on
`require_main_study()` and on the coverage record existing, then writes into
`<run_dir>/analysis/` (override with `--out`):

- `confirmatory_family.json` — §4.5.6's C1–C4, raw and Holm-adjusted p-values, plus the item accounting behind each test
- `beta_qe_intervals.json` — §4.5.5's per-model conditional-logit β_QE and J = −β_QE intervals
- `truncation_rates.json` — §4.4's truncated-generation rate by precision
- `analysis_manifest.json` — the interval method, the coverage check that fixed it, library versions, the input run's identity and the code commit

The confirmatory model, the bf16→nf4 contrast, the detector-to-slot mapping and
the item sets are module constants in `run_analysis.py`, deliberately not CLI
flags.

## Running the full test suite

```bash
pytest
```

GPU-free throughout: some tests hit the network for real (LiveCodeBench/
evalplus dataset downloads) and run real sandboxed code execution (no model
needed), but nothing here needs a GPU or model weights.

## Syncing validation/main artifacts down from the H100

```bash
scripts/sync_from_h100.sh <ssh-alias> [<remote-repo-path>] -- --dry-run   # preview first
scripts/sync_from_h100.sh <ssh-alias>                                     # then the real sync
```
