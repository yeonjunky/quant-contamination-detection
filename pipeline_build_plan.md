# Data-Collection Pipeline: Build Plan

> **BUILD STATUS 2026-09-18 — the analysis layer, the decoding freeze and the validation
> boundary are implemented.** This supersedes the items the 2026-09-09 note below lists as
> "pre-execution work", each of which now exists in code:
>
> - **§4.5.6's confirmatory family.** `src/qcd/analysis/confirmatory.py` runs C1–C3 (paired
>   t-tests per detector) and C4 (the probability-family-minus-CDD AUC reversal test, with a
>   DeLong covariance for the paired comparison), applies Holm across the four slots, and
>   retains a not-estimable slot at p=1 rather than dropping it.
>   `src/qcd/analysis/study_inputs.py` reads the raw tree into exactly those arrays and
>   reports what it dropped. `scripts/run_analysis.py` is the entry point; the model, the
>   contrast, the detector-to-slot mapping and the item sets are module constants there, not
>   CLI flags.
> - **§4.5.5's Q2 interval.** `src/qcd/analysis/conditional_logit.py` fits an item-stratified
>   conditional logistic model per model, one quantized level against bf16 at a time, and
>   reports the β_QE Wald interval and J = −β_QE. `mixed_effects.py`'s variational-Bayes fit
>   remains available for point estimates and variance components, but its mean-field
>   posterior SD is explicitly not the reported interval.
>   `src/qcd/analysis/coverage.py` and `scripts/verify_interval_coverage.py` run the
>   synthetic-data coverage check that fixes which method supplies that interval;
>   `run_analysis.py` refuses to run without its record
>   (`pipeline/analysis_artifacts/interval_coverage_check.json`).
> - **The §4.6 validation boundary, enforced in code.** Every manifest carries a required
>   `study_phase` (`engineering_validation` / `main_study`), and
>   `io/manifest.py::require_main_study` is the first thing `run_analysis.py` calls, before
>   it opens a parquet file. Smoke tests write to `data/raw/validation/...`; `run_main.py`
>   writes to `data/raw/main`. The smoke tests no longer print or store per-item pass rates
>   or detector scores — they check only range, finiteness and schema — and both now take
>   `--model`/`--quant` so any roster arm can be validated before the main run reaches it.
>   `pilot/aggregate.py`'s path that computed required item counts from observed effects is
>   deleted; planning numbers come from `analysis/power.py` with the paper's own illustrative
>   effect sizes.
> - **§4.4's frozen decoding settings.** `constants.py` holds them and `models/loader.py`
>   hands them to `generate()` as a `GenerationConfig` object (`top_p=1.0`, top-k disabled
>   via `top_k=0`, `repetition_penalty=1.0`, no length penalty or minimum-length constraint,
>   512-token cap); the adapter additionally replaces each checkpoint's own
>   `generation_config` with one carrying nothing but its stop/pad token ids, because the
>   knobs whose only "off" value is `None` cannot be pinned. Stored per-token
>   log-probabilities are the raw values from `outputs.logits`, read before the logits
>   processors run.
> - **§4.3's AWQ calibration, fixed rather than selected.** `scripts/quantize_model.py` has
>   no `--calibration` argument and no compare-then-copy-the-winner step: every model is
>   quantized with the same code calibration set and written straight to the canonical
>   `data/quantized/<model>-awq/`. It records the selected-row hashes and the shuffle seed,
>   and writes `calibration_overlap_report.json`, without which `models/loader.py` refuses to
>   load the checkpoint.
> - **scipy** is now named explicitly in `requirements-local.txt` and `requirements-h100.txt`,
>   because `analysis/confirmatory.py` and `analysis/conditional_logit.py` import it directly
>   rather than relying on statsmodels to keep pulling it in.

> **PROTOCOL CLARIFICATION 2026-09-09.** The canonical manuscript §4.4–§4.5.6 governs
> scoring and inference. CDD's short-output normalization is corrected and versioned in the
> main-run manifest. Existing AUC/GLMM helpers do not constitute a completed C1–C4 report:
> the six-AUC C4 reversal test, its fixed Qwen2.5-32B item sets, Q2 diagnostics/credible-interval
> reporting, and AWQ calibration-overlap review remain pre-execution work. Earlier claims of
> a complete analysis layer refer to the helpers available then, not this full analysis protocol.

> **DESIGN CHANGE 2026-08-28 — scientific pilot removed.** Local dry runs and bounded H100
> smoke tests are engineering validation only. Their outputs cannot be used as manuscript evidence,
> effect-size or power inputs, CDD screening, detector ranking, or confirmatory-test eligibility.
> `run_main.py` is the only study-data driver. The former `run_pilot.py` and
> `aggregate_pilot.py` entrypoints now intentionally refuse execution; older status notes below are
> retained as implementation history and are superseded wherever they describe a scientific pilot.

> **DESIGN CHANGE 2026-08-05 — this plan predates a model-roster change.** Llama-3.3-70B and
> Gemma-4-31B-it were removed from the design entirely; Llama-3.1-8B-Instruct was added
> (see `paper/revision_provenance.md`, 2026-08-05 entry). This document has been updated to
> remove the resulting 70B/multi-GPU execution path. No arm exceeds 32.5B, the whole ladder
> fits a single H100/H200. The registry in `pipeline/src/qcd/models/registry.py` is the authoritative roster.

> **BUILD STATUS 2026-08-14 — the mock-verifiable scope of this plan is built.** Every module
> below reachable without a CUDA GPU exists and is tested (data loaders, generation, scoring
> incl. real sandboxed code execution, all three detectors, the full analysis layer, the then-current
> pilot gate/report later retired by the 2026-08-28 change, io writers, `scripts/run_dry_run.py`/
> `run_pilot.py`/`run_main.py`/
> `sync_from_h100.sh`). `models/loader.py`'s real `generate()`/`score_logprobs()`, the
> GPTQ/AWQ backend, and `scripts/run_smoke_test.py` remain deferred to a session on an actual
> CUDA machine (this build pass ran on a Mac, Apple Silicon, no CUDA — confirmed with the user
> before starting). See `pipeline/README.md`'s "What's built vs. what's deferred" for current
> status; this document's file tree below is otherwise still the accurate target design.
>
> **BUILD STATUS 2026-08-15 — real GPU path validated on H100.** `models/loader.py`'s real
> `generate()`/`score_logprobs()` implemented for bf16/bnb-int8/bnb-nf4 and validated end-to-end
> by the now-written `scripts/run_smoke_test.py` (Qwen2.5-7B-Instruct, BNB-nf4, 5 real HumanEval
> items, real H100: peak GPU memory 6.69GB, all checklist items passing). `requirements-h100.txt`
> is now a pinned lockfile (previously an unpinned placeholder), captured from this run
> (`pipeline/envs/local-smoke-freeze.txt`). Open items unchanged from before this pass except as
> noted: the GPTQ/AWQ backend is still deferred (open assumption #1, below, still unresolved —
> not attempted this pass, out of its agreed scope). One real finding surfaced by running on real
> hardware: HumanEval+/MBPP+ candidate-code assembly (`real_run.py`'s `_assemble_candidate_code`)
> assumed a raw code continuation, but the -Instruct roster answers in prose + a markdown code
> fence, so the assembled candidate for these two conditions was not runnable Python at all —
> the sandbox was scoring the prose, not the solution. Fixed
> in the same session — markdown-fence stripping + evalplus's own `sanitize`/`code_extract`
> post-processing, re-verified on the same real HumanEval completions. Per §4.6 the
> validation run establishes only that extraction produces runnable candidate code; no pass
> rate from it is recorded. Applied uniformly
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
> the two 32B models are deliberately not quantized in this pass, left for the frozen main
> run. A code-domain vs. general-chat calibration comparison was also run that day (motivated
> by §2.7/§4.3's literature review). It was engineering work on a handful of validation items,
> so it supports no comparative claim and none is carried forward; §4.3 now fixes the code
> calibration set for every model with no selection step, and `scripts/quantize_model.py` no
> longer produces the chat variant (see the 2026-09-18 status block above). Real
> finding: AWQ checkpoints use ~15-16GB peak GPU memory through plain transformers, not the
> ~4-5GB on-disk size would suggest (known compressed-tensors/transformers rough edge with
> asymmetric zero-points, vllm-project/llm-compressor#1550) — worth watching for the 32B arms'
> memory headroom later. `requirements-h100.txt` re-pinned again: `pip install llmcompressor`
> silently upgraded torch/transformers/numpy mid-session (re-verified the bnb path still passes
> under the new versions). Full detail: `pipeline_implementation_log.md`'s 2026-08-15 entry, §7.

## Historical context at initial planning

At the time this plan was first written, the repo held only the paper draft, review history, and reference CSVs —
no code, scripts, or environment file existed then (confirmed at that time:
`git status` clean, no `.py`/`.ipynb`/`requirements.txt` anywhere, no venv/conda on
that machine). The paper's own committed execution order (AGENTS.md §7 / paper draft
§5) has 9 steps, but nothing has been built to run any of them. The user wants a plan
to actually start collecting data, split across two machines: **spike tests here**
(RTX 4060 laptop GPU, 8GB VRAM) and **the actual main experiment on an
already-provisioned H100 SSH box**. This plan is the engineering scaffolding needed to
execute the paper's existing 9-step plan — it does not change the experimental design,
which is fixed by AGENTS.md §5's invariants (log-odds scale for Q2, no HumanEval/MBPP+
pooling, Gemma excluded from main analysis, observational not causal, etc.).

Decisions already confirmed with the user:
- H100 access: an already-provisioned SSH box (not ephemeral cloud rental) — environment
  setup happens once and persists.
- Result sync: rsync/scp over SSH, manual or scripted, back to this repo/machine.
- Local spike scope: dry-run first (synthetic/mocked outputs, validates code logic,
  zero GPU/downloads), then a real small-scale smoke test if it fits (Qwen2.5-7B in
  BNB-nf4, ~4-5GB, should fit in 8GB) to catch real quantization/loading issues the mock
  can't.
- Framework: Hugging Face `transformers` + `bitsandbytes` (int8/nf4) + llm-compressor AWQ
  — chosen because it maps 1:1 onto the paper's own quantization-ladder terminology.

**Important correction found during planning:** the libraries initially considered for
the calibration-based arm — `AutoGPTQ` and `AutoAWQ` — are both archived/unmaintained
(AutoGPTQ archived April 2025; AutoAWQ archived ~May 2025; HF `transformers` dropped
AutoGPTQ backend support). The maintained replacements that produce the same output
formats are **GPTQModel** (GPTQ) and **llm-compressor** (AWQ). The implemented and frozen
experimental condition uses llm-compressor AWQ uniformly; GPTQ is follow-up work only.

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
    constants.py             # single source for α, power target, CDD separation-reduction assumption —
                              #   imported everywhere, never re-typed (AGENTS.md §3.3 discipline)
    config.py                # ModelSpec / QuantSpec / DatasetSpec / RunConfig dataclasses
    models/
      registry.py            # mirrors paper's model table 1:1 (kept in sync manually)
      loader.py               # load_model(spec, quant) — branches bf16/bnb-int8/bnb-nf4/
                              #   gptq-awq-int4 compatibility enum (AWQ only, see
                              #   engineering decision #1)/mock
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
      logodds.py                   # the named invariant (AGENTS.md §5 point 3)
      auc.py                        # paired AUC + Hanley-McNeil SE, numpy-only (matches the
                                     #   "scipy 없음" convention already used for AGENTS.md's numbers)
      aggregation.py                 # HARD-FAILS if HumanEval+MBPP+ combined into one cell
      power.py                        # pre-execution planning/sensitivity calculations
      mixed_effects.py                 # correct ~ precision*exposure_proxy + (1|item) + (1|model);
                                        #   VB point estimates/variance components, NOT the interval
      confirmatory.py                    # §4.5.6's C1-C4 + DeLong covariance + Holm
      conditional_logit.py                # §4.5.5's reported β_QE Wald interval, per model
      coverage.py                          # §4.5.5's synthetic interval-coverage check
      study_inputs.py                       # raw-tree -> the arrays §4.4/§4.5.5/§4.5.6 need
      _stats.py                              # the no-scipy primitives (normal CDF via math.erf,
                                              #   its inverse by bisection); scipy is used only
                                              #   for the C1-C4 / Wald tail probabilities
    pilot/                              # legacy package name; development-only diagnostics
      pilot_report.py                   # synthetic validation diagnostics, not study estimates
      aggregate.py                       # development-only descriptive summary; the
                                          #   observed-effect sizing path was removed
    io/
      raw_writer.py                     # item-level raw data writer
      manifest.py                        # per-run manifest: study_phase, git commit, config
                                          #   hash, lib versions, HF model revision hashes,
                                          #   seeds, timestamps; also the §4.6 study-phase gate,
                                          #   §4.3's resolved library defaults and the AWQ
                                          #   calibration-overlap report
  analysis_artifacts/                    # tracked analysis inputs/records (not run outputs)
    interval_coverage_check.json          # §4.5.5's coverage record, read by run_analysis.py
  tests/                                   # actual file names, not the planned ones: the
                                            #   invariants below live in these modules
    test_logodds.py                        # regression test against paper §4.5.3's worked table
                                            #   (β=0.50 → 5.6pp/7.7pp/−2.2pp — verified against draft)
    test_aggregation.py                    # the HumanEval+MBPP+ pooling guard
    test_pilot.py                          # validation-diagnostics regression tests
    test_power.py                          # reproduces §4.5.2's SE(AUC) values
    test_detectors.py                      # CDD / perplexity / Min-k% known-answer tests
    test_io.py                             # raw-schema round trip
    test_confirmatory.py / test_conditional_logit.py / test_coverage.py /
      test_run_analysis.py                 # §4.5.5-§4.5.6 analysis layer
    test_study_phase.py                    # the §4.6 validation/main-study boundary
    test_decoding_settings.py              # §4.4's pinned decoding settings
    test_calibration_records.py            # §4.3's AWQ calibration + overlap records
    test_mock_pipeline_end_to_end.py       # the dry-run harness (see below)
  scripts/
    run_dry_run.py                          # local mock spike, interactive
    run_smoke_test.py                        # real HumanEval smoke test, any model/precision
    run_lcb_smoke_test.py                     # real LiveCodeBench stdin/functional smoke test
    quantize_model.py                          # one-time offline AWQ quantization (§4.3)
    run_pilot.py / aggregate_pilot.py           # retired compatibility entrypoints; exit without running
    run_main.py                                  # H100 main-experiment driver
    run_analysis.py                               # frozen analysis driver (§4.5.5/§4.5.6)
    verify_interval_coverage.py                    # §4.5.5's pre-main-run coverage check
    search_olmo_corpus.py / manage_olmo_pretraining_scan.py   # Olmo corpus-reference work
    sync_from_h100.sh                              # rsync wrapper
data/                                               # gitignored
  raw/
    validation/smoke_test/<quant>/                  # manifest.json (engineering_validation) + raw/
    validation/lcb_smoke_test/                      # manifest.json + lcb_smoke_report.json + raw/
    main/                                           # manifest.json (main_study),
                                                    #   resolved_library_defaults.json,
                                                    #   raw/*.parquet, cache/, analysis/
  quantized/<model>-awq/                            # AWQ checkpoint + quantization_manifest.json
                                                    #   + calibration_overlap_report.json
```

Each run owns its own manifest, generation cache and analysis outputs under its run
directory — `real_run.py` writes `<output_dir>/manifest.json`, `<output_dir>/raw/`,
`<output_dir>/cache/` and `<output_dir>/resolved_library_defaults.json`, and
`run_analysis.py` defaults to `<run_dir>/analysis/`. There is no separate top-level
`data/cache/` or `data/manifests/` tree.

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
  llm-compressor are CUDA-runtime sensitive, and a silent version mismatch is worse than a
  crash for numbers that need to be trustworthy. (GPTQModel was evaluated but not used —
  see "Open assumptions" #1.)
- `HF_HOME`/`HF_HUB_CACHE` pointed at large disk **outside** the git repo. The largest
  retained arm is 32.5B and its bf16 baseline fits one H100, tightly.
- Save `pip freeze` output as a committed reproducibility artifact per environment.

## Local spike tests (this machine)

**1. Dry-run (mock, no GPU/downloads):**
`models/mock.py` implements the same interface real loaders expose, so the rest of the
pipeline never branches on mock-vs-real. Use a real small tokenizer (e.g. GPT-2's,
CPU-only) for realistic token ids, but synthetic logits/text. The mock injects a known
generative process so end-to-end tests can assert implementation invariants rather than
only "didn't crash." Run ~10-20 fake items covering all four conditions through the
validation path, checking schema integrity, finite scores, the log-odds worked example,
and the HumanEval+MBPP+ pooling guard. Synthetic diagnostics are explicitly marked
validation-only and cannot become study estimates. Ship as both a pytest test and an
interactive script.

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

Mapped onto the paper's own step numbering (AGENTS.md §7):

| Step | What | Notes |
|---|---|---|
| (pre-step) | Env setup + pinned lockfile + implementation checks | Validate loading, schema, finite log-probabilities, sandboxing, memory, and throughput without using outcome values as study evidence or design inputs |
| 1-2 | Continuous scoring + detector scoring pipelines, at scale | Built/tested locally first (dry-run + smoke); CDD's multi-sample cost shares generations with step 1 via the cache |
| 3 | Count LCB pre/post items (≥1,000 target each) | The common boundary is 2025-01-01: Olmo 3's official model cards state a Dec. 2024 cutoff, later than Qwen2.5's 2024-09-19 release bound. Current release_v6 count: pre 873 / post 182 (availability check, not an experiment result). |
| 4 | Cutoff verification | Llama-3.1 is externally verified; Qwen2.5 uses its official release date; both Olmo Instruct model cards state Dec. 2024. Month-level conservatism makes 2025-01-01 the first eligible common post-cutoff day. Direct Olmo corpus search supplies confirmed-positive evidence, not verified negative labels. |
| 5 | TRACER positive-evidence search (+ Olmo3 corpus-reference search) | Storage was verified before the exhaustive scan: the pinned 7B pretraining manifest is 3.23 TB compressed and the project filesystem had about 290 TB free. Search pretraining and post-training stages separately; report `confirmed-match`, `no-match-found`, and `not-observable` without estimating *e*, false-positive rate, or false-negative rate. |
| 6 | Engineering validation only | Run local dry-run and bounded H100 smoke tests; inspect only loading, compatibility, schemas, finite values, sandboxing, memory, and throughput. Written to `data/raw/validation/...` with `study_phase="engineering_validation"`, which the analysis entry point refuses. |
| 7 | Freeze operational configuration | Hardware/runtime failures observed before outcome inspection may change operational settings; document and freeze them. C1–C4, item sets, detector priority, and sample planning remain unchanged. |
| 8 | Full run, the only study-data source | `scripts/run_main.py` — all retained 7B/8B/32B bf16 baselines and quantized arms run on one H100, writing `data/raw/main` with `study_phase="main_study"`; store all item-level rows. |
| 9 | Analysis (Q1a/Q1b/Q2) | `scripts/run_analysis.py <run-dir>`. No GPU needed once `data/raw/` is synced back. Run `scripts/verify_interval_coverage.py` first: §4.5.5's coverage record fixes which method supplies the reported β_QE interval, and the driver refuses to run without it. `require_main_study` rejects a validation tree before any parquet file is opened. |

## Raw data schema + sync

Parquet, one row per measurement (not per aggregate) — required for the paired (Q1a)
and mixed-effects (Q2, `(1|item)+(1|model)`) analyses; aggregate-only storage would
foreclose both:
- `items.parquet` — item/dataset metadata (id, dataset, condition, difficulty bucket,
  legacy dataset-level condition, release/version pins).
- `model_item_labels.parquet` — one row per (model, item), with the frozen primary and
  optional sensitivity temporal boundaries, `possible-exposure` / `clean-by-model-cutoff` /
  `shared-clean-control`, and `boundary_ambiguous`. Q1b/Q2 join this table rather than a
  global item label.
- Corpus-reference outputs — method-specific `confirmed-match`, `no-match-found`, or
  `not-observable` plus coverage metadata; they are not merged into temporal proxy labels.
- `generations.parquet` (plus `generations.<part>.parquet` for each flushed batch) — one row
  per (model, quant, item, sample): generated text, full per-token logprob array (Min-k%
  needs the actual lowest-k% subset, not a scalar), partial pass rate, test results, decoding
  params, model/tokenizer revision hashes. §4.4 adds `truncated_at_cap` and `max_new_tokens`
  (so the truncated-generation rate can be reported by precision) and a
  `decoding_settings_id` pointing at the manifest's full resolved decoding record. The greedy
  row additionally carries what §4.4 requires next to a probability-detector score: the
  fixed prompt's per-token logprob array, the scored text's sha256, its character span and
  token indices in the rendered prompt, whether a chat template was applied, and the
  `chat_template_id`.
- `chat_templates.parquet` — one row per (model, quant, chat template), holding the rendered
  template text itself. The template is identical for every item of a given tokenizer, so it
  is stored once per run and generations rows carry only its `chat_template_id`.
- `detector_scores.parquet` (same `.<part>` convention) — one row per
  (model, quant, item, detector): score, threshold used, source sample ids.
- Validation diagnostics — stored outside the study-data namespace and marked
  `development_only_not_manuscript_evidence`; never committed as citation-relevant results.
- `manifest.json` per run — `study_phase` (required: `engineering_validation` or
  `main_study`), git commit, config hash, library versions, model revision hashes, seeds,
  machine id, timestamps, plus §4.4's resolved decoding settings for the greedy and sample
  passes and their ids. `resolved_library_defaults.json` sits next to it with §4.3's
  "left at the library default" values as they resolved per model/precision.

`scripts/sync_from_h100.sh` wraps `rsync -avz --progress <ssh-alias>:<repo>/data/raw/
./data/raw/`, excluding model-weight caches. Run with `--dry-run` first; verify row
counts / checksums match the H100-side manifest after transfer.

## Testing/verification

Full pytest suite runs GPU-free (mock-only), so it can run in CI on every push. Tests
tied to named invariants, each checked against the actual paper draft (not re-derived
from memory):
- `test_logodds.py` — regression vs. §4.5.3's table (β=0.50 → 5.6pp/7.7pp/−2.2pp) —
  **verified against `paper/paper_draft.md` line 540 during this planning session.**
- `test_aggregation.py` — HumanEval+MBPP+ combination raises `PooledSecondaryConditionsError`.
- `test_pilot.py` — verifies synthetic validation-diagnostics construction only; no CDD gate or study eligibility.
- `test_power.py` — reproduces §4.5.2's SE(AUC) values, numpy-only.
- `test_detectors.py` — known-answer tests on synthetic logprob arrays (CDD, perplexity, Min-k%).
- `test_confirmatory.py` — the paired test checked end to end against `scipy.stats.ttest_rel`;
  the C4 reversal construction and the Holm correction over the four slots.
- `test_conditional_logit.py` / `test_coverage.py` / `test_run_analysis.py` — §4.5.5's β_QE
  interval, its synthetic coverage check, and the analysis driver's two gates.
- `test_study_phase.py` — a validation tree is refused by `require_main_study`.
- `test_decoding_settings.py` — §4.4's pinned settings survive a checkpoint's own
  `generation_config`.
- `test_calibration_records.py` — §4.3's calibration manifest and overlap report, and the
  loader's refusal without one.
- `test_io.py` (raw-schema round trip), `test_mock_pipeline_end_to_end.py`.
The real GPU smoke test stays a manual checklist in `pipeline/README.md` (no GPU CI
runner). Passing it authorizes only that the implementation is ready to start the frozen main run.

## Engineering decisions and remaining assumptions

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
   workaround (`pipeline_implementation_log.md`'s 2026-08-15 entry, §7). Paper §4.3 now
   freezes AWQ-int4 as the uniform calibration-based 4-bit condition, not a per-model design axis.

   `models/loader.py` retains the compatibility enum name `Quant.GPTQ_AWQ_INT4`. The paper
   names this condition AWQ-int4, but **the recorded value is the enum's own string,
   `gptq_awq_int4`** — that is what `config.py`'s `Quant.GPTQ_AWQ_INT4` holds, what
   `real_run.py` writes into the manifest's `quant_levels`, what every generations /
   detector-scores row carries in its `quant` column, and what
   `scripts/quantize_model.py` writes into `quantization_manifest.json`. Anything reading the
   raw tables or the manifests has to match on `gptq_awq_int4`, not on "AWQ-int4". GPTQ itself
   was never attempted, not ruled out on technical merit for the non-Olmo3 arms.

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
4. **LiveCodeBench snapshot pinning** — resolved: the loader pins `release_v6`; the common
   2025-01-01 boundary was counted as pre 873 / post 182 / total 1,055.
5. **Torch/bitsandbytes CUDA tier** (12.x vs 13.0) — re-check at actual install time.
6. **Olmo3 corpus storage** — resolved for the current environment: the pinned 7B mix is
   3.23 TB compressed versus about 290 TB free on the project filesystem. Streaming remains
   resumable so the corpus need not be materialized twice.
7. **TRACER implementation availability** — no public executable release was found; the
   paper-specified routing, prompts, embedding adapter, schema, and orchestration have been
   reimplemented locally. Fidelity validation against Olmo evidence remains required.

## Verification (end-to-end)

1. `pytest pipeline/tests/` passes fully on this machine with the mock-only profile
   installed (no GPU library present) — proves the pipeline logic is sound independent
   of hardware.
2. `python pipeline/scripts/run_dry_run.py` produces a small validation-only tree and
   synthetic implementation diagnostics that are not study estimates.
3. `python pipeline/scripts/run_smoke_test.py` on this machine loads real Qwen2.5-7B nf4,
   completes without OOM/NaN, and produces output matching the mock's schema.
4. On the H100: loading, schema, finite-value, sandbox, memory, and throughput checks pass
   before the operational configuration is frozen; outcome values are not used for design decisions.
5. `scripts/sync_from_h100.sh --dry-run` then a real sync round-trips a small validation
   file correctly before the first main-run sync.
6. `python pipeline/scripts/verify_interval_coverage.py` runs before the main run and leaves
   `pipeline/analysis_artifacts/interval_coverage_check.json` in place — it is CPU-only and
   touches no study data, and `run_analysis.py` will not run without it.
7. `python pipeline/scripts/run_analysis.py <run-dir>` on the synced main-study tree writes
   `confirmatory_family.json`, `beta_qe_intervals.json`, `truncation_rates.json` and
   `analysis_manifest.json`. Pointing it at a validation tree must fail with the
   `require_main_study` error rather than produce numbers.
