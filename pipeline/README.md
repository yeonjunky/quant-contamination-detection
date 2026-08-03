# pipeline/

Data-collection pipeline for the quantization contamination-detection paper.
See `../pipeline_build_plan.md` at the repo root for the full design and
rationale; this file is just setup + run instructions.

## Environments

Two machines, three install profiles:

- **This machine (RTX 4060, 8GB VRAM)** — mock-only profile for the dry run,
  optionally layered with the real-smoke profile for the nf4 smoke test.
- **H100 box** — full pinned GPU stack for the actual pilot/main experiment.

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

Manual go/no-go gate before spending H100 time. Run
`scripts/run_smoke_test.py` and confirm all of the following:

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
