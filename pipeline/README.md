# pipeline/

Data-collection pipeline for the quantization contamination-detection paper.
See `../pipeline_build_plan.md` at the repo root for the full design and
rationale; this file is just setup + run instructions.

## What's built vs. what's deferred

Everything reachable without a CUDA GPU is built and tested: dataset
loaders (LiveCodeBench/HumanEval+/MBPP+), generation sampling + cache,
sandboxed scoring (real subprocess execution, not mocked), all three
detectors (CDD/perplexity/Min-k% Prob), the full statistical analysis layer
(power tables, log-odds base-rate model, mixed-effects GLMM), the pilot gate
+ report, raw-data/manifest writers, and the mock dry run
(`scripts/run_dry_run.py`) exercising all of it end-to-end.

**Built and validated on real H100 hardware (2026-08-15):**
`models/loader.py`'s real `generate()`/`score_logprobs()` for the fp16 and
bitsandbytes (int8/nf4) backends, exercised end-to-end by
`scripts/run_smoke_test.py` against Qwen2.5-7B-Instruct/BNB-nf4 — real
quantized load (peak 6.69GB), real sampling, real sandboxed code execution,
real detector scoring, real raw-data writer, all checklist items passing.
`scripts/run_pilot.py`/`run_main.py` now work all the way through for the
fp16/bnb quant levels (`Quant.FP16`/`BNB_INT8`/`BNB_NF4`).

**Real finding from that run, fixed same session (2026-08-15):**
`real_run.py`'s `_assemble_candidate_code()` used to assume HumanEval+/MBPP+
candidate code was `item.prompt + completion_text` verbatim, but the roster
is entirely -Instruct models queried through a chat template — the model
answers with prose plus a fenced code block, not a raw continuation, so
`partial_pass_rate` came back 0.0 across the board for these two datasets
(confirmed on 5 real HumanEval items). Fixed by stripping the markdown fence
first (`_strip_markdown_fence`) and reusing evalplus's own post-processing
(`evalplus.sanitize.sanitize`/`code_extract` — the same step evalplus's
leaderboard runs on LLM output) rather than reimplementing extraction; see
`_assemble_candidate_code`'s docstring for the HumanEval+/MBPP+-vs-LCB
extractor choice. Re-scoring the same 5 real completions afterward: 4/5 went
from 0.0 to 1.0, the 5th stayed near-zero (0.02) from a genuine model logic
error, not an extraction failure. Applied uniformly to LCB too, but that
side is **not yet empirically validated against a real LCB completion** —
only HumanEval was exercised on real hardware this session.

**Still deferred:** the GPTQ/AWQ backend (`models/loader.py`'s
`_load_gptq_or_awq`) — which model arm uses GPTQ vs AWQ and which quantized
checkpoint to use is an open, unresolved design question
(`pipeline_build_plan.md` open assumption #1), not yet implementation work.

## Environments

Machines and install profiles this design targets:

- **A CUDA laptop (e.g. RTX 4060, 8GB VRAM)** — mock-only profile for the
  dry run, optionally layered with the real-smoke profile for the nf4 smoke
  test.
- **H100 box (validated 2026-08-15)** — `requirements-h100.txt` is now a
  pinned lockfile for the fp16/bnb stack (torch 2.6.0+cu124, transformers
  5.15.0, bitsandbytes 0.50.1, accelerate 1.14.0, ...), confirmed working via
  `scripts/run_smoke_test.py`. `gptqmodel`/`llmcompressor` remain commented
  out — see "What's built" above.
- **A Mac (Apple Silicon, no CUDA)** — mock-only profile plus the real,
  GPU-free pieces (dataset downloads, sandboxed code execution, statistics):
  everything under "Running the dry run" and the full `pytest` suite below.

```bash
# from pipeline/
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-local.txt        # mock-only, no GPU library at all
pip install -e .

# only when running against a real GPU backend (smoke test or H100 pilot/main):
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-smoke.txt        # local laptop-scale smoke test
# or: pip install -r requirements-h100.txt   # full pinned H100 stack
```

## Local smoke-test checklist (Qwen2.5-7B, BNB-nf4)

**Run and passing (2026-08-15, real H100).** `scripts/run_smoke_test.py`
runs this as an automated checklist against 5 real HumanEval items and exits
non-zero if any item fails — the numbers below are from that run:

- [x] nf4 load fits in a plausible few-GB band — peak 6.69GB (script checks 2-12GB, no OOM on the 80GB card)
- [x] teacher-forced logprob scoring returns finite values (no NaN/-inf)
- [x] repeated T=0.8 samples for the same item actually differ
- [x] the real sandboxed code-execution path runs generated code correctly (runs — see the README's "What's built" note on `partial_pass_rate` currently coming back 0.0 for HumanEval+/MBPP+ for an unrelated, separate reason: markdown-fenced chat output, not a sandbox execution failure)
- [x] CDD / perplexity / Min-k% detectors run end-to-end without dtype/shape errors and produce plausible values
- [x] the real raw-data writer's output matches the mock's schema exactly
- [x] per-item wall-clock is sane — model load 10.7s (weights cached), ~10-19s/item
- [x] `pip freeze` saved to `envs/local-smoke-freeze.txt`

## Running the dry run

```bash
python scripts/run_dry_run.py
```

Exercises the full step 1-9 code path on ~10-20 synthetic items with zero
GPU/downloads. Also runnable as `pytest tests/test_mock_pipeline_end_to_end.py`.

## Running the full test suite

```bash
pytest
```

GPU-free throughout: some tests hit the network for real (LiveCodeBench/
evalplus dataset downloads) and run real sandboxed code execution (no model
needed), but nothing here needs a GPU or model weights.

## Syncing pilot/main results down from the H100

```bash
scripts/sync_from_h100.sh <ssh-alias> [<remote-repo-path>] -- --dry-run   # preview first
scripts/sync_from_h100.sh <ssh-alias>                                     # then the real sync
```
