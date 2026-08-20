# pipeline/

Data-collection pipeline for the quantization contamination-detection paper.
See `../pipeline_build_plan.md` at the repo root for the full design and
rationale; this file is just setup + run instructions.

## What's built vs. what's deferred

The following GPU-free components are built and tested: dataset
loaders (LiveCodeBench/HumanEval+/MBPP+), generation sampling + cache,
sandboxed scoring (real subprocess execution, not mocked), all three
detectors (CDD/perplexity/Min-k% Prob), the full statistical analysis layer
(power tables, log-odds base-rate model, mixed-effects GLMM), the pilot gate
+ report, raw-data/manifest writers, and the mock dry run
(`scripts/run_dry_run.py`) exercising all of it end-to-end.

**Built and validated on real H100 hardware (2026-08-15):**
`models/loader.py`'s real `generate()`/`score_logprobs()` for the bf16 and
bitsandbytes (int8/nf4) backends, exercised end-to-end by
`scripts/run_smoke_test.py` against Qwen2.5-7B-Instruct/BNB-nf4 — real
quantized load (peak 6.69GB), real sampling, real sandboxed code execution,
real detector scoring, real raw-data writer, all checklist items passing.
`scripts/run_pilot.py`/`run_main.py` now work all the way through for the
bf16/bnb quant levels (`Quant.BF16`/`BNB_INT8`/`BNB_NF4`).

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
error, not an extraction failure. The LCB path was subsequently validated
on four real Qwen2.5-7B BNB-NF4 outputs spanning pre/post and
stdin/functional tasks; see `scripts/run_lcb_smoke_test.py`.

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

Code-domain and general-chat calibration data were compared by quantizing
Qwen2.5-7B-Instruct twice (code-domain:
`flytech/python-codes-25k`; chat: `HuggingFaceH4/ultrachat_200k`) and
compared on the same 5-item smoke test. Both passed 4/5 items; the fifth had
a model error, and detector scores were similar. This sample is insufficient
to compare calibration domains. The code-domain variant is stored at
`data/quantized/<model>-awq/`; the chat-domain comparison checkpoint stays
on disk as `-awq-chat` for a future re-comparison at real pilot scale.

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
models (Qwen2.5-32B, Olmo3.1-32B) — deliberately deferred to the real
pilot/main run, not attempted in this validation pass.

## Environments

Machines and install profiles this design targets:

- **A CUDA laptop (e.g. RTX 4060, 8GB VRAM)** — mock-only profile for the
  dry run, optionally layered with the real-smoke profile for the nf4 smoke
  test.
- **H100 box (validated 2026-08-15)** — `requirements-h100.txt` is now a
  pinned lockfile covering bf16/bnb *and* GPTQ/AWQ (llm-compressor):
  torch 2.13.0+cu130, transformers 5.14.1, bitsandbytes 0.50.1,
  accelerate 1.14.0, llmcompressor 0.13.0, compressed-tensors 0.18.0, ...
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
pip install -r requirements-local.txt        # mock-only, no GPU library at all
pip install -e .

# only when running against a real GPU backend (smoke test or H100 pilot/main):
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-smoke.txt        # local laptop-scale smoke test
# or: pip install -r requirements-h100.txt   # full pinned H100 stack
```

## Local smoke-test checklist

**Run and passing (2026-08-15, real H100), against bf16/bnb *and* AWQ.**
`scripts/run_smoke_test.py` runs this as an automated checklist against 5
real HumanEval items and exits non-zero if any item fails —
`--model`/`--quant`/`--checkpoint-path` select which backend/checkpoint (see
the script's own docstring for examples, including pointing it at one of
`quantize_model.py`'s calibration-comparison checkpoints directly). Numbers
below are from the default bnb-nf4 run; the AWQ runs (Qwen2.5-7B ×2
calibration variants, Olmo3-7B ×1) all passed the same checklist too, with
peak memory in the ~15-16GB band documented above instead:

- [x] quantized load fits in a plausible memory band (quant-dependent — see `PLAUSIBLE_PEAK_GB` and the AWQ memory finding above) — bnb-nf4 peak 6.69GB, no OOM on the 80GB card
- [x] teacher-forced logprob scoring returns finite values (no NaN/-inf)
- [x] repeated T=0.8 samples for the same item actually differ
- [x] the real sandboxed code-execution path runs generated code correctly (runs — see the README's "What's built" note on `partial_pass_rate` currently coming back 0.0 for HumanEval+/MBPP+ for an unrelated, separate reason: markdown-fenced chat output, not a sandbox execution failure)
- [x] CDD / perplexity / Min-k% detectors run end-to-end without dtype/shape errors and produce plausible values
- [x] the real raw-data writer's output matches the mock's schema exactly
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

Exercises the full step 1-9 code path on ~10-20 synthetic items with zero
GPU/downloads. Also runnable as `pytest tests/test_mock_pipeline_end_to_end.py`.

## Running and aggregating the pilot

```bash
python scripts/run_pilot.py --lcb-cutoff 2025-01-01
```

After writing item-level parquet, the driver creates `pilot_summary.json`
and `power_recompute.json`. To rebuild only those summaries without loading
models or regenerating answers:

```bash
python scripts/aggregate_pilot.py data/raw/pilot
```

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
