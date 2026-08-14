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
(`scripts/run_dry_run.py`) exercising all of it end-to-end. `scripts/
run_pilot.py`/`run_main.py` orchestrate the same pipeline against real
models — everything in them works up through model loading.

**Deferred to a session on an actual CUDA machine:** `models/loader.py`'s
real `generate()`/`score_logprobs()` bodies (currently `NotImplementedError`
stubs), the GPTQ/AWQ backend, and `scripts/run_smoke_test.py` — this
repo was built on a Mac (Apple Silicon, no CUDA), so those paths could be
written but not exercised or verified here; see `pipeline_build_plan.md`.

## Environments

Machines and install profiles this design targets:

- **A CUDA laptop (e.g. RTX 4060, 8GB VRAM)** — mock-only profile for the
  dry run, optionally layered with the real-smoke profile for the nf4 smoke
  test (not yet runnable — see "What's built" above).
- **H100 box** — full pinned GPU stack for the actual pilot/main experiment.
- **This machine (Mac, Apple Silicon)** — mock-only profile plus the real,
  GPU-free pieces (dataset downloads, sandboxed code execution, statistics):
  everything under "Running the dry run" and the full `pytest` suite below.

```bash
# from pipeline/
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-local.txt        # mock-only, no GPU library at all
pip install -e .

# only when running the real nf4 smoke test:
pip install -r requirements-smoke.txt
```

`requirements-h100.txt` is an **unpinned placeholder** — it must become a
fully version-pinned lockfile against the H100's actual driver/CUDA before
real use. See the comments in that file.

## Local smoke-test checklist (Qwen2.5-7B, BNB-nf4)

**Not yet runnable — `scripts/run_smoke_test.py` doesn't exist yet** (see
"What's built vs. what's deferred" above; needs a CUDA machine). Once
written, this is the manual go/no-go gate before spending H100 time —
confirm all of the following:

- [ ] nf4 load fits in ~4-5GB, no OOM on the 8GB card
- [ ] teacher-forced logprob scoring returns finite values (no NaN/-inf)
- [ ] repeated T=0.8 samples for the same item actually differ
- [ ] the real sandboxed code-execution path runs generated code correctly
- [ ] CDD / perplexity / Min-k% detectors run end-to-end without dtype/shape errors and produce plausible values
- [ ] the real raw-data writer's output matches the mock's schema exactly
- [ ] per-item wall-clock is sane (smell test only)
- [ ] `pip freeze` saved to `envs/local-smoke-freeze.txt`

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
