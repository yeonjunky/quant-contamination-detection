# Olmo training-data corpus-reference search

This CPU-only first pass for paper section 5, step 5 searches released Olmo
training rows for normalized-verbatim prompt occurrences and token n-gram
overlap. Its `corpus_status` is method-family evidence, not a verified Q1b
contamination label. Surface/AST matching, semantic/paraphrase matching, and the
TRACER reimplementation remain separate later stages; their outputs must not be
averaged into this score.

## One corpus per checkpoint, not one "Olmo3 corpus"

The design has two Olmo arms, and they do **not** share a bulk pretraining mix.
A result therefore belongs to one checkpoint: a `confirmed-match` in the 7B mix
says nothing about the 32B arm, and a `no-match-found` in one mix is not a
`no-match-found` in the other. Paper section 4.2 stores corpus evidence on a
per-model axis for this reason.

Every scan entry point takes a required `--model`, every evidence row carries
it, and the checkpoint/repository pairing is checked against
`src/qcd/ground_truth/olmo_corpora.py`, which refuses a pairing it records as
belonging to the other arm.

## Verified public inputs (2026-08-21; revisions and post-training assignment 2026-10-19)

| Checkpoint | Stage | Hugging Face repository | Pinned scan revision | Hub size / rows | Text-bearing schema |
|---|---|---|---|---:|---|
| `Olmo3-7B-Instruct` | bulk pretraining | [`allenai/dolma3_mix-6T-1025-7B`](https://huggingface.co/datasets/allenai/dolma3_mix-6T-1025-7B) | `2ca900fbe14e86c5c83d064d9f0882f1c0b8c05b` | 3.23 TB compressed at the pinned scan revision (47,025 shards); card: 23.7 TB uncompressed, 3.87B documents | `id`, `text`, `metadata`, `source`, `version`, `created`, `added`, `doc`, `attributes` |
| `Olmo3.1-32B-Instruct` | bulk pretraining | [`allenai/dolma3_mix-6T`](https://huggingface.co/datasets/allenai/dolma3_mix-6T) | `689a3ea2d8217e64d73a5058913fa43ad15e81aa` | 4.41 TB compressed at the pinned revision (63,913 files) | same JSONL.Zstandard schema |
| both Instruct arms | SFT | [`allenai/Dolci-Instruct-SFT`](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT) | `bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221` | 3.06 GB download; 2,152,112 rows | `id`, nested `messages`, `source_dataset`, `domain` |
| both Instruct arms | DPO | [`allenai/Dolci-Instruct-DPO`](https://huggingface.co/datasets/allenai/Dolci-Instruct-DPO) | `aed155cf32e809b590490b6c3577ee4b0d0a5019` | 810 MB download; 259,922 rows | nested `chosen` and `rejected`, model IDs, `prompt_id`, `preference_type` |
| both Instruct arms | RL/RLVR prompt mixture | [`allenai/Dolci-Instruct-RL`](https://huggingface.co/datasets/allenai/Dolci-Instruct-RL) | `1a8d79b1012d77e4bb1fd7f699c5c7eb33ef931e` | 483 MB download; 169,964 rows | `prompt`, `solution`, `ground_truth`, `source_prompt`, outputs and source metadata |

How the two values that were open until 2026-10-19 were settled (Hub API and
model cards, checked that day):

- **32B pretraining revision.** `689a3ea2…` is the current Hub commit. Its 63,913
  data files have the same paths, LFS oids and sizes as at `6612d71…`, the commit
  that finished the 2025-12-16/17 upload; every later commit edits only the
  README. `6612d71…` itself is not used, because its README config points
  `datasets` at a single directory. The `Olmo-3-1125-32B` card names
  `allenai/dolma3_mix-5.5T-1125`, which now redirects to this repository. The
  Olmo 3 report notes that the released mixes are reconstructions of the data
  sampled in training (arXiv:2512.13961 v2, p.14, footnote 5).
- **Post-training assignment.** Both arms' model cards name the same three
  repositories: `Olmo-3-7B-Instruct-SFT` and `Olmo-3.1-32B-Instruct-SFT` name
  `Dolci-Instruct-SFT`; the 7B and `Olmo-3.1-32B-Instruct-DPO` cards name
  `Dolci-Instruct-DPO`; `Olmo-3.1-32B-Instruct` and the 7B card name
  `Dolci-Instruct-RL`. The report's artifact list pairs both Instruct models with
  `Dolci-Instruct-{SFT|DPO|RL}`, while Think data is split into 7B and 32B
  repositories. Neither source says the 32B run used exactly the revisions
  pinned here: SFT and DPO were re-uploaded on 2026-02-03 and 2026-02-20 with
  unchanged row counts (pre-upload revisions `51b25163…` and `9d9461c8…`), and
  the report's training tables describe only the 7B Instruct runs.

Sizes above are availability checks, not experimental results. Compressed sizes
were summed from Hugging Face Hub API file manifests; row counts and
post-training sizes come from each repository's `dataset_info`. Pin the Hub
revision in a scientific run's manifest. The 7B card warns that some olmOCR
science-PDF texts were redacted after training and marked `[REMOVED]`; therefore
a negative search is not proof of non-exposure for those documents.

Hugging Face `datasets` supports `streaming=True`, so the scanner does not first
download a whole dataset. Streaming still transfers every scanned shard. A
complete 3–4 TB compressed pretraining pass is therefore an expensive fallback,
not an index. The practical sequence is:

1. run and validate all three post-training repositories (about 4.35 GB total);
2. run source-targeted pretraining shards, especially `stack_edu`, with a
   persistent shard manifest;
3. verify whether an existing indexed service covers these exact, revision-pinned
   mixes; otherwise build a resumable n-gram/FM index near the corpus storage.

## Command

Run from `pipeline/` with the project environment active. `--max-documents` is
only a wiring check and must never be interpreted as a negative corpus label.

```bash
python scripts/search_olmo_corpus.py \
  --benchmark all \
  --model Olmo3-7B-Instruct \
  --hf-repo allenai/Dolci-Instruct-SFT \
  --stage sft \
  --max-documents 1000 \
  --output ../data/olmo_ground_truth/sft_humaneval_smoke.jsonl
```

`--model` is required and is written into every evidence row. The pair is
checked against `olmo_corpora.py`: each post-training repository is recorded
for both Instruct arms, so a post-training scan runs once per arm with that
arm's `--model`. Every item keeps up to 20 ranked candidate rows with document
text (`--candidates-per-item`); `corpus_status` describes one candidate, and an
item is a `confirmed-match` when any of its rows is.

`--benchmark all` combines HumanEval, MBPP+, LCB-pre, and LCB-post before
building the benchmark-side index, so each training corpus is transferred and
scanned once rather than four times.

### Resumable pretraining scan

Do not use one repository-wide `load_dataset` stream for the multi-terabyte
pretraining mixes. Initialize a revision-pinned SQLite manifest once, then run
the resumable worker:

One manifest covers one checkpoint's mix. `--model` is stored in the manifest
metadata and re-checked on every `run`, so two checkpoints' shards cannot end up
in the same manifest.

```bash
python scripts/manage_olmo_pretraining_scan.py \
  --manifest ../data/olmo_ground_truth/pretraining/olmo3_7b_manifest.sqlite \
  init \
  --model Olmo3-7B-Instruct \
  --repo allenai/dolma3_mix-6T-1025-7B \
  --revision 2ca900fbe14e86c5c83d064d9f0882f1c0b8c05b

python scripts/manage_olmo_pretraining_scan.py \
  --manifest ../data/olmo_ground_truth/pretraining/olmo3_7b_manifest.sqlite \
  run \
  --model Olmo3-7B-Instruct \
  --repo allenai/dolma3_mix-6T-1025-7B \
  --revision 2ca900fbe14e86c5c83d064d9f0882f1c0b8c05b \
  --output-dir ../data/olmo_ground_truth/pretraining/olmo3_7b_shards
```

The 32B arm uses its own manifest, its own output directory, and
`--model Olmo3.1-32B-Instruct --repo allenai/dolma3_mix-6T`. Its `--revision`
has to be supplied explicitly, because no scan revision is pinned for that mix
here.

`status` prints completed shard, byte, and document counts. Multiple `run`
processes may share one manifest: each claim records a worker ID, a unique
lease token, and a heartbeat,
and completion is accepted only from the current owner. The default stale lease
timeout is 600 seconds. A surviving worker returns an expired lease to `pending`
before claiming its next shard, so a terminated worker's shard is retried
without rescanning completed work. Per-worker output directories prevent a late
stale process from overwriting the current owner's result. `--retry-failed`
schedules each currently failed shard for one additional attempt, and
`--keep-going` lets the worker continue after a failure.
Software-related source directories have queue priority, but all manifest
shards remain required for an exhaustive result. Each completed shard writes
only nonzero string evidence, atomically, while the manifest records negative
scan coverage and the document count. This avoids storing 1,597 negative rows
for every one of tens of thousands of shards. The pretraining worker retains
the top 20 candidates per benchmark item and shard by default. Each candidate
includes the full and normalized source text, SHA-256, selected source metadata,
matched token position, an n-gram example, and local context so downstream
TRACER stages do not require a second multi-terabyte corpus pass. Change the
bound explicitly with `--candidates-per-item`; it is part of the retrieval
configuration and must be frozen before the scientific run.

After `status` reports that every shard is complete, `finalize --output PATH`
reduces the shard evidence to the globally ranked candidate bound (20 rows
per benchmark item by default). It refuses to
produce an apparently complete result while any shard is pending, running, or
failed.

For deterministic local validation, `--jsonl PATH` accepts JSONL rows and
recursively extracts string fields from plain-text or nested chat schemas. The
output keeps model, corpus, stage, document ID, scan count, exact flag, n-gram
coverage, threshold, `match_detected`, `corpus_status`, and `coverage_complete`
in every benchmark-item row. `confirmed-match` requires positive evidence;
`no-match-found` is emitted only after a complete requested scan; bounded smoke
or shard scans use `not-observable` until exhaustive finalization.

### Frozen retrieval settings (2026-10-19, before any main-study output was read)

- **`confirmed-match` rule:** normalized verbatim containment, or token 13-gram
  coverage `>= 0.8` within one document (`n=13`, coverage `0.8`).
- **Candidate bound:** 20 candidates per benchmark item, per shard and after
  `finalize` (`--candidates-per-item 20`). A larger bound than the earlier 5
  keeps room for distinct documents when one problem text is mirrored across
  many pages; candidates below the match rule still go on to TRACER.
- **Negative-control count, reported with every pretraining result:** LCB-post
  items (contest date on or after 2025-01-01) cannot be in either Olmo bulk
  pretraining mix, whose crawl ends 2024-12-31 (arXiv:2512.13961). The number of
  LCB-post items that reach `confirmed-match` in a pretraining scan is reported
  as the rule's observed false-match count. It does not re-tune the rule.

- **Sensitivity rule, reported beside the primary one:** token 8-gram coverage
  `>= 0.5` (or verbatim containment), the rule the Olmo 3 report gives for its
  own post-training decontamination (8-gram matching, 0.5 overlap threshold).
  The same pass computes it, every row carries `sensitivity_ngram_coverage` and
  `sensitivity_corpus_status`, and candidates rank primary-match first, then
  sensitivity-match, then coverage. It never replaces the primary status. In a
  post-training mix, a sensitivity match is a case Olmo's own filter would have
  removed.

The primary rule and the candidate bound are retrieval choices of this study,
not settings taken from TRACER or from the Olmo 3 decontamination pipeline.

The resumable manifest records evidence schema version 5: version 3 added the
tri-state output, version 4 added the `model` on the manifest metadata and on
every evidence row, and version 5 added the sensitivity rule's fields. A
manifest at an older schema version must be reinitialized rather than mixed
with new shard rows; `finalize` refuses a schema-3 manifest (its rows name no
checkpoint) and any manifest not at version 5.

## Not implemented in this first pass

- edit-distance and AST-based surface/program matching;
- embedding retrieval and semantic/paraphrase adjudication;
- TRACER's three LLM stages and validation against open-data evidence;
- an indexed search service that avoids transferring every pretraining shard;
- descriptive comparison of separate method families and TRACER on confirmed
  positive evidence.

These omissions are explicit because exact/n-gram evidence alone is a lower
bound on contamination and cannot establish a clean label. Even after the
additional method families are run, `no-match-found` is not verified
non-exposure; therefore this workflow reports `confirmed-match`,
`no-match-found`, and `not-observable` and does not identify proxy-label error
rate *e*, false-positive rate, or false-negative rate.
