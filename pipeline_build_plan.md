# Data-Collection Pipeline: Build Plan

> **DESIGN CHANGE 2026-08-05 — this plan predates a model-roster change.** Llama-3.3-70B and
> Gemma-4-31B-it were removed from the design entirely; Llama-3.1-8B-Instruct was added
> (see `paper/revision_provenance.md`, 2026-08-05 entry). Consequences for this document:
> every 70B fp16 / multi-GPU-rental reference below is obsolete (no arm exceeds 32.5B; the
> whole ladder fits a single H100/H200), the HF cache sizing note shrinks accordingly, and
> `pilot_report.py` must compute **five** pilot quantities (a)–(e) per paper §4.7 (the
> "(a)-(d)" below is stale). The registry in `pipeline/src/qcd/models/registry.py` is the
> authoritative roster.

> **BUILD STATUS 2026-08-14 — the mock-verifiable scope of this plan is built.** Every module
> below reachable without a CUDA GPU exists and is tested (data loaders, generation, scoring
> incl. real sandboxed code execution, all three detectors, the full analysis layer, pilot
> gate/report, io writers, `scripts/run_dry_run.py`/`run_pilot.py`/`run_main.py`/
> `sync_from_h100.sh`). `models/loader.py`'s real `generate()`/`score_logprobs()`, the
> GPTQ/AWQ backend, and `scripts/run_smoke_test.py` remain deferred to a session on an actual
> CUDA machine (this build pass ran on a Mac, Apple Silicon, no CUDA — confirmed with the user
> before starting). See `pipeline/README.md`'s "What's built vs. what's deferred" for current
> status; this document's file tree below is otherwise still the accurate target design.
>
> **BUILD STATUS 2026-08-15 — real GPU path validated on H100.** `models/loader.py`'s real
> `generate()`/`score_logprobs()` implemented for fp16/bnb-int8/bnb-nf4 and validated end-to-end
> by the now-written `scripts/run_smoke_test.py` (Qwen2.5-7B-Instruct, BNB-nf4, 5 real HumanEval
> items, real H100: peak GPU memory 6.69GB, all checklist items passing). `requirements-h100.txt`
> is now a pinned lockfile (previously an unpinned placeholder), captured from this run
> (`pipeline/envs/local-smoke-freeze.txt`). Open items unchanged from before this pass except as
> noted: the GPTQ/AWQ backend is still deferred (open assumption #1, below, still unresolved —
> not attempted this pass, out of its agreed scope). One real finding surfaced by running on real
> hardware: HumanEval+/MBPP+ candidate-code assembly (`real_run.py`'s `_assemble_candidate_code`)
> assumed a raw code continuation, but the -Instruct roster answers in prose + a markdown code
> fence, so `partial_pass_rate` came back 0.0 for these two conditions on the first pass. Fixed
> in the same session — markdown-fence stripping + evalplus's own `sanitize`/`code_extract`
> post-processing, re-verified against the same 5 real HumanEval completions (4/5: 0.0 -> 1.0;
> the 5th's near-zero score is a genuine model error, not an extraction bug). Applied uniformly
> to LCB too, but that side is not yet validated against a real LCB completion — see
> `pipeline_implementation_log.md`'s 2026-08-15 entry for detail. `pipeline/README.md`'s "What's
> built vs. what's deferred" has the current authoritative status.
>
> **BUILD STATUS 2026-08-15 (same day, continued) — the GPTQ_AWQ_INT4 quant rung is
> implemented and validated, via AWQ only — GPTQ itself was not implemented.** "Open
> assumption #1" below resolved differently than originally planned: llm-compressor (AWQ)
> uniformly for all five roster models, not GPTQModel, not a per-model GPTQ/AWQ split —
> GPTQModel's own architecture registry has no `olmo3` entry (checked against source), while
> llm-compressor confirmed working on Olmo3-7B-Instruct with no workaround. GPTQModel
> remains a possible follow-up (see "Open assumptions" #1's "worth trying later" note), not
> ruled out on technical merit for the non-Olmo3 arms. New `scripts/quantize_model.py`
> does the one-time offline quantization (quantize-once-and-save, unlike bnb's
> load-time quantization); `models/loader.py`'s `_load_gptq_or_awq` loads the result. Validated
> on Qwen2.5-7B-Instruct (×2 calibration variants) and Olmo3-7B-Instruct (×1) — Llama-3.1-8B and
> the two 32B models are deliberately not quantized in this pass, left for the real pilot/main
> run. Also ran a live comparison of code-domain vs. general-chat calibration data (motivated by
> §2.7/§4.3's literature review) — no detectable difference at n=5 (too small to resolve either
> way); canonicalized the code-domain variant, consistent with the original hypothesis. Real
> finding: AWQ checkpoints use ~15-16GB peak GPU memory through plain transformers, not the
> ~4-5GB on-disk size would suggest (known compressed-tensors/transformers rough edge with
> asymmetric zero-points, vllm-project/llm-compressor#1550) — worth watching for the 32B arms'
> memory headroom later. `requirements-h100.txt` re-pinned again: `pip install llmcompressor`
> silently upgraded torch/transformers/numpy mid-session (re-verified the bnb path still passes
> under the new versions). Full detail: `pipeline_implementation_log.md`'s 2026-08-15 entry, §8.

## Context

The repo currently holds only the paper draft, review history, and reference CSVs —
no code, no scripts, no environment file, nothing executable exists yet (confirmed:
`git status` clean, no `.py`/`.ipynb`/`requirements.txt` anywhere, no venv/conda on
this machine). The paper's own committed execution order (CLAUDE.md §7 / paper draft
§5) has 9 steps, but nothing has been built to run any of them. The user wants a plan
to actually start collecting data, split across two machines: **spike tests here**
(RTX 4060 laptop GPU, 8GB VRAM) and **the actual pilot/main experiment on an
already-provisioned H100 SSH box**. This plan is the engineering scaffolding needed to
execute the paper's existing 9-step plan — it does not change the experimental design,
which is fixed by CLAUDE.md §5's invariants (log-odds scale for Q2, no HumanEval/MBPP+
pooling, Gemma excluded from main analysis, observational not causal, etc.).

Decisions already confirmed with the user:
- H100 access: an already-provisioned SSH box (not ephemeral cloud rental) — environment
  setup happens once and persists.
- Result sync: rsync/scp over SSH, manual or scripted, back to this repo/machine.
- Local spike scope: dry-run first (synthetic/mocked outputs, validates code logic,
  zero GPU/downloads), then a real small-scale smoke test if it fits (Qwen2.5-7B in
  BNB-nf4, ~4-5GB, should fit in 8GB) to catch real quantization/loading issues the mock
  can't.
- Framework: Hugging Face `transformers` + `bitsandbytes` (int8/nf4) + GPTQ/AWQ tooling
  — chosen because it maps 1:1 onto the paper's own quantization-ladder terminology.

**Important correction found during planning:** the exact libraries the user named for
the GPTQ/AWQ arms — `AutoGPTQ` and `AutoAWQ` — are both archived/unmaintained
(AutoGPTQ archived April 2025; AutoAWQ archived ~May 2025; HF `transformers` dropped
AutoGPTQ backend support). The maintained replacements that produce the same output
*formats* (still called GPTQ-int4 / AWQ-int4, matching the paper's terminology exactly)
are **GPTQModel** (GPTQ) and **llm-compressor** (AWQ). This plan uses those instead of
the archived libraries — flagging it explicitly since it's a deviation from what was
literally requested, even though it doesn't touch the paper's argument structure at all.

## Repo layout

Two new top-level directories:

- `pipeline/` — tracked in git. Real engineering effort, no secrets, worth versioning
  and reviewing like any code.
- `data/` — gitignored. Raw item-level outputs and model-generation caches, synced down
  from the H100. This follows the repo's existing precedent of gitignoring large
  artifacts (`*.pdf`) — same reasoning applies to raw experiment output.

The `.gitignore` currently has a preemptive `.commandcode` line from an earlier commit,
but nothing was ever put there and it's a hidden, semantically-empty name. Recommend
dropping it in favor of the plain `pipeline/`/`data/` split above, consistent with this
repo's existing visible, descriptive directory-naming convention (`paper/`, `reference/`,
`review/`, `figures/`).

```
pipeline/
  pyproject.toml
  README.md                  # setup + run instructions + the local smoke-test checklist
  requirements-local.txt     # mock-only profile (no torch/transformers/bitsandbytes needed)
  requirements-h100.txt      # full pinned GPU stack
  src/qcd/
    constants.py             # single source for CDD_GATE_AUC=0.7936, α, power target —
                              #   imported everywhere, never re-typed (CLAUDE.md §3.3 discipline)
    config.py                # ModelSpec / QuantSpec / DatasetSpec / RunConfig dataclasses
    models/
      registry.py            # mirrors paper's model table 1:1 (kept in sync manually)
      loader.py               # load_model(spec, quant) — branches fp16/bnb-int8/bnb-nf4/
                              #   gptqmodel/llm-compressor-awq/mock
      mock.py                # same interface as loader.py, deterministic synthetic outputs
    data/
      schema.py               # canonical Item dataclass
      livecodebench.py        # HF dataset load, pre/post-cutoff split, pins release_version
      humaneval.py / mbppplus.py   # via evalplus; verify pinned counts are 164 / 378
    generation/
      sampler.py               # 1 greedy + n T=0.8 samples, every precision
      cache.py                 # content-addressed cache shared between scoring & detector steps
    scoring/
      pass_rate.py             # partial test-case pass rate
      logprob.py                # teacher-forced per-token logprob
      sandbox.py                 # subprocess code execution, timeouts, no network
    detectors/
      cdd.py                     # verify exact formula against Dong et al. 2024 /
                                  #   arXiv:2603.03203 replication before implementing —
                                  #   do not guess it
      perplexity.py
      mink_prob.py                # needs full per-token logprob array, not a summary scalar
      threshold.py                 # ξ handling; guard against re-selecting ξ on eval set
    analysis/
      logodds.py                   # the named invariant (CLAUDE.md §5 point 3)
      auc.py                        # paired AUC + Hanley-McNeil SE, numpy-only (matches the
                                     #   "scipy 없음" convention already used for CLAUDE.md's numbers)
      aggregation.py                 # HARD-FAILS if HumanEval+MBPP+ combined into one cell
      power.py                        # power recompute from pilot values
      mixed_effects.py                 # correct ~ precision*contaminated + (1|item) + (1|model)
    pilot/
      cdd_gate.py                      # gate check against constants.CDD_GATE_AUC (0.7936)
      pilot_report.py                   # computes pilot quantities (a)-(d) from paper §4.7
    io/
      raw_writer.py                     # item-level raw data writer
      manifest.py                        # per-run manifest: git commit, config hash, lib
                                          #   versions, HF model revision hashes, seeds, timestamps
  tests/
    test_logodds.py                      # regression test against paper §4.5.3's worked table
                                          #   (β=0.50 → 5.6pp/7.7pp/−2.2pp — verified against draft)
    test_pooling_guard.py
    test_cdd_gate.py                      # parametrized against §4.6's table, gate = 0.7936
                                          #   (verified against draft: break-even ≈0.79)
    test_auc_hanley_mcneil.py             # reproduces §4.5.2's SE(AUC) values
    test_mink_prob.py / test_perplexity.py
    test_raw_schema_roundtrip.py
    test_mock_pipeline_end_to_end.py       # the dry-run harness (see below)
  scripts/
    run_dry_run.py                          # local mock spike, interactive
    run_smoke_test.py                        # local real Qwen2.5-7B nf4 smoke test
    run_pilot.py                              # H100 pilot driver
    run_main.py                                # H100 main-experiment driver
    sync_from_h100.sh                          # rsync wrapper
data/                                           # gitignored
  raw/{pilot,main}/...
  cache/generations/...
  manifests/...
```

## Environment setup

**Local (RTX 4060, 8GB VRAM, python3.10.12, driver reports CUDA 13.2, no toolkit/conda
installed):**
- No CUDA toolkit install needed — `torch`/`bitsandbytes` wheels bundle their own CUDA
  runtime; the driver only needs to support that runtime (it does — 13.2 is backward
  compatible with older runtime tiers).
- Plain `python3 -m venv`, upgrade pip first (current pip is old).
- Two install profiles: **mock-only** (`numpy`, `pandas`, `pyarrow`, `pytest`, `evalplus`,
  `datasets`/`huggingface_hub` for metadata — no GPU library at all) for the dry run, and
  **real-smoke** (adds `torch`, `transformers`, `bitsandbytes`, `accelerate`) only when
  running the real nf4 smoke test.
- Recommend a `torch` build from the CUDA 12.4/12.6 wheel index (best-tested tier with
  bitsandbytes currently) rather than the newest 13.0 tier. The 4060 (Ada Lovelace,
  compute capability 8.9) is comfortably inside bitsandbytes' supported range either way.

**H100 (already provisioned, persists across runs):**
- Confirm its own driver/CUDA compatibility independently — don't assume it matches the
  laptop.
- Fully pinned `requirements-h100.txt` (exact versions, not ranges) — `bitsandbytes`/
  GPTQModel/llm-compressor are CUDA-runtime sensitive, and a silent version mismatch is
  worse than a crash for numbers that need to be trustworthy.
- `HF_HOME`/`HF_HUB_CACHE` pointed at large disk **outside** the git repo (weights run up
  to ~141GB for Llama-3.3-70B fp16).
- The one-time multi-GPU rental for the 70B fp16 pass reuses the same lockfile to avoid a
  third compatibility surface; its window must cover the complete pass (CDD multi-sample
  generation + logprob scoring), not generation alone.
- Save `pip freeze` output as a committed reproducibility artifact per environment.

## Local spike tests (this machine)

**1. Dry-run (mock, no GPU/downloads):**
`models/mock.py` implements the same interface real loaders expose, so the rest of the
pipeline never branches on mock-vs-real. Use a real small tokenizer (e.g. GPT-2's,
CPU-only) for realistic token ids, but synthetic logits/text. The mock must inject a
*known* generative process — a hidden contaminated/clean flag and quality parameter per
synthetic item — producing peakier/more-repeatable completions and higher-confidence
logprobs for "contaminated" items, and a fractional (not just 0/1) partial pass rate.
This lets the end-to-end test assert **expected-signed** outputs, not just "didn't
crash." Run ~10-20 fake items covering all four conditions (LCB-pre/post-style,
HumanEval-style, MBPP+-style) through the full step 1→9 code path, asserting the three
named invariants directly: log-odds transform matches the paper's worked table, pooling
HumanEval+MBPP+ raises an error, and the CDD gate check classifies correctly against
0.7936. Ship as both a pytest test (CI-runnable, no GPU) and an interactive script.

**2. Real smoke test (Qwen2.5-7B, BNB-nf4, if the dry-run passes):**
Deliberately tiny — a loading/integration check, not a scientific run. 5-10 HumanEval
items (shortest prompts, no LCB dataset-versioning complexity), 1 greedy + 2 T=0.8
samples per item. Must verify before declaring H100-readiness: nf4 load actually fits
in ~4-5GB; logprob scoring returns finite values (a real numerical failure mode the mock
can't surface); repeated samples actually differ (catches deterministic-seeding bugs);
the real sandboxed code-execution path runs correctly (mock bypasses this entirely);
detectors run end-to-end without dtype/shape errors and produce plausible values; the
real writer's output matches the mock's schema exactly. Save `pip freeze` here too, to
cross-check against the H100 lockfile later.

## H100 execution order

Mapped onto the paper's own step numbering (CLAUDE.md §7):

| Step | What | Notes |
|---|---|---|
| (pre-step) | Env setup + pinned lockfile + known-answer check | e.g. Qwen2.5-7B fp16 pass@1 roughly matches its public HumanEval score — real-hardware sanity gate before trusting anything downstream |
| 1-2 | Continuous scoring + detector scoring pipelines, at scale | Built/tested locally first (dry-run + smoke); CDD's multi-sample cost shares generations with step 1 via the cache |
| 3 | Count LCB pre/post items (≥1,000 target each) | No GPU needed; do a provisional count now, re-confirm after step 4 verifies actual cutoffs |
| 4 | LLMLagBench cutoff verification | Needs all 5 models loaded; keep on H100 for environment consistency |
| 5 | TRACER residual contamination (+ Olmo3 corpus ground-truth search) | Confirm H100 disk can hold the Olmo3 pretraining corpus before assuming it fits — not yet verified. Parallelizable with step 4 |
| 6 | Pilot: Qwen2.5-7B + Olmo3-7B, BNB-nf4 | **Check the CDD gate (≥0.7936) before trusting any Q1b pilot number** — if it fails, fall back to probability-based detectors only for the rest of the pilot |
| 7 | Recompute power from pilot values | Pure computation, local or H100; decide final main-experiment n |
| 8 | Full run, store all item-level raw data | H100 for 7B/32B; 70B fp16 needs the separate one-time multi-GPU rental (or the pre-specified int8-anchored fallback, reported as a limitation exactly as the paper already does) |
| 9 | Analysis (Q1a/Q1b/Q2) | No GPU needed once `data/raw/` is synced back |

## Raw data schema + sync

Parquet, one row per measurement (not per aggregate) — required for the paired (Q1a)
and mixed-effects (Q2, `(1|item)+(1|model)`) analyses; aggregate-only storage would
foreclose both:
- `items.parquet` — item/dataset metadata (id, dataset, condition, difficulty bucket,
  contamination proxy label, TRACER label, release/version pins).
- `generations.parquet` — one row per (model, quant, item, sample): generated text, full
  per-token logprob array (Min-k% needs the actual lowest-k% subset, not a scalar),
  partial pass rate, test results, decoding params, model/tokenizer revision hashes.
- `detector_scores.parquet` — one row per (model, quant, item, detector): score,
  threshold used, source sample ids.
- `pilot_summary.json` / `power_recompute.json` — small, **committed to git** like
  `figures/*.png` (citation-relevant numeric artifacts).
- `manifest.json` per run — git commit, library versions, model revision hashes, seeds,
  machine id, timestamps.

`scripts/sync_from_h100.sh` wraps `rsync -avz --progress <ssh-alias>:<repo>/data/raw/
./data/raw/`, excluding model-weight caches. Run with `--dry-run` first; verify row
counts / checksums match the H100-side manifest after transfer.

## Testing/verification

Full pytest suite runs GPU-free (mock-only), so it can run in CI on every push. Tests
tied to named invariants, each checked against the actual paper draft (not re-derived
from memory):
- `test_logodds.py` — regression vs. §4.5.3's table (β=0.50 → 5.6pp/7.7pp/−2.2pp) —
  **verified against `paper/paper_draft.md` line 540 during this planning session.**
- `test_pooling_guard.py` — HumanEval+MBPP+ combination raises `PooledSecondaryConditionsError`.
- `test_cdd_gate.py` — parametrized against §4.6's table, gate = 0.7936 — **verified
  against the draft's "break-even point is a baseline AUC of ≈0.79" (line 599).**
- `test_auc_hanley_mcneil.py` — reproduces §4.5.2's SE(AUC) values, numpy-only.
- `test_mink_prob.py` / `test_perplexity.py` — known-answer tests on synthetic logprob arrays.
- `test_raw_schema_roundtrip.py`, `test_mock_pipeline_end_to_end.py`.
The real GPU smoke test stays a manual checklist in `pipeline/README.md` (no GPU CI
runner) — the explicit go/no-go gate before spending H100 time.

## Open assumptions to confirm before/during implementation

1. **GPTQModel / llm-compressor substitution** for the archived AutoGPTQ/AutoAWQ —
   confirm before locking `requirements-h100.txt`.

   **Resolved 2026-08-15 — AWQ only, via llm-compressor, uniformly for all five models. GPTQ
   was not implemented.** The original plan here assumed *both* libraries, GPTQModel for a
   GPTQ arm and llm-compressor for an AWQ arm, with the split decided per model. That was
   dropped, not deferred: GPTQModel's own architecture registry
   (`gptqmodel/models/auto.py`, fetched and read directly from source) has no `olmo3` entry
   — `olmo2` maps to `LlamaQModel` (a Llama clone), but `olmo`/`olmo3` are absent — so
   GPTQModel would very likely fail on the two Olmo3 roster arms without upstream support.
   llm-compressor has no per-architecture registry at all — its `AWQModifier`/
   `QuantizationModifier` recipe targets any HF-loadable causal LM's `nn.Linear` layers by
   name pattern, and was confirmed empirically to work on Olmo3-7B-Instruct with no
   workaround (`pipeline_implementation_log.md`'s 2026-08-15 entry, §8). Paper §4.3 treats
   "GPTQ-int4 or AWQ-int4" as interchangeable representatives of one calibration-based
   4-bit condition, not a per-model design axis, so using AWQ uniformly is more consistent
   with that framing than a per-model GPTQ/AWQ split would have been, not less.

   `models/loader.py`'s `Quant.GPTQ_AWQ_INT4` — the paper's own combined name for this
   quantization rung — is therefore backed by AWQ only. GPTQ itself was never attempted, not
   ruled out on technical merit for the non-Olmo3 arms.

   **Worth trying later:** GPTQModel on the non-Olmo3 arms (Qwen2.5, Llama-3.1) specifically,
   if a true GPTQ-vs-AWQ technique comparison becomes useful, or if GPTQModel gains Olmo3
   support upstream and a uniform-technique run becomes possible again. Also worth
   revisiting if AWQ's plain-transformers memory overhead (README's "Real finding" —
   ~15-16GB peak for a 7B model, not the ~4-5GB the on-disk size suggests, a known
   compressed-tensors/transformers asymmetric-zero-point rough edge,
   vllm-project/llm-compressor#1550) turns out to matter at 32B scale — GPTQModel's own
   inference path may not have the same overhead.
2. **CDD's exact statistic** must be pulled verbatim from Dong et al. 2024 / cross-checked
   against arXiv:2603.03203's replication when `detectors/cdd.py` is written — not guessed.
3. **evalplus version** yielding exactly 164/378 items must be pinned and confirmed — different
   releases have shipped different MBPP+ subset sizes historically.
4. **LiveCodeBench snapshot pinning** — assumed the HF dataset supports a `release_version`
   pin; not yet directly inspected.
5. **Torch/bitsandbytes CUDA tier** (12.x vs 13.0) — re-check at actual install time.
6. **Olmo3 corpus storage** for TRACER's ground-truth search — not yet verified to fit on
   the H100 box's disk.
7. **No confirmed public code release found** for TRACER (arXiv:2605.24079) or LLMLagBench
   (arXiv:2511.12116) in a quick check — may need reimplementation from the papers'
   methodology sections; worth a dedicated check before steps 4/5 begin.

## Verification (end-to-end)

1. `pytest pipeline/tests/` passes fully on this machine with the mock-only profile
   installed (no GPU library present) — proves the pipeline logic is sound independent
   of hardware.
2. `python pipeline/scripts/run_dry_run.py` produces a small `data/raw/` tree and a
   pilot-quantities report from purely synthetic data, with expected-signed results.
3. `python pipeline/scripts/run_smoke_test.py` on this machine loads real Qwen2.5-7B nf4,
   completes without OOM/NaN, and produces output matching the mock's schema.
4. On the H100: the known-answer check (real fp16 pass@1 vs. public score) passes before
   any pilot data is trusted.
5. `scripts/sync_from_h100.sh --dry-run` then a real sync round-trips a small test file
   correctly before the first real pilot sync.
