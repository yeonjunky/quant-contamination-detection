# Olmo training-data ground-truth search

This CPU-only first pass for paper section 5, step 5 searches released Olmo
training rows for normalized-verbatim prompt occurrences and token n-gram
overlap. Its `string_match_label` is method-family evidence, not the final Q1b
ground-truth label. Surface/AST matching, semantic/paraphrase matching, and the
TRACER reimplementation remain separate later stages; their outputs must not be
averaged into this score.

## Verified public inputs (2026-08-21)

| Model stage | Hugging Face repository | Hub size / rows | Text-bearing schema |
|---|---|---:|---|
| 7B bulk pretraining | [`allenai/dolma3_mix-6T-1025-7B`](https://huggingface.co/datasets/allenai/dolma3_mix-6T-1025-7B) | 3.35 TB compressed in the current Hub file manifest; card: 23.7 TB uncompressed, 3.87B documents | `id`, `text`, `metadata`, `source`, `version`, `created`, `added`, `doc`, `attributes` |
| 32B bulk pretraining | [`allenai/dolma3_mix-6T`](https://huggingface.co/datasets/allenai/dolma3_mix-6T) | 4.41 TB compressed in the current Hub file manifest | same JSONL.Zstandard schema |
| SFT | [`allenai/Dolci-Instruct-SFT`](https://huggingface.co/datasets/allenai/Dolci-Instruct-SFT) | 3.06 GB download; 2,152,112 rows | `id`, nested `messages`, `source_dataset`, `domain` |
| DPO | [`allenai/Dolci-Instruct-DPO`](https://huggingface.co/datasets/allenai/Dolci-Instruct-DPO) | 810 MB download; 259,922 rows | nested `chosen` and `rejected`, model IDs, `prompt_id`, `preference_type` |
| RL/RLVR prompt mixture | [`allenai/Dolci-Instruct-RL`](https://huggingface.co/datasets/allenai/Dolci-Instruct-RL) | 483 MB download; 169,964 rows | `prompt`, `solution`, `ground_truth`, `source_prompt`, outputs and source metadata |

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
  --hf-repo allenai/Dolci-Instruct-SFT \
  --stage sft \
  --max-documents 1000 \
  --output ../data/olmo_ground_truth/sft_humaneval_smoke.jsonl
```

`--benchmark all` combines HumanEval, MBPP+, LCB-pre, and LCB-post before
building the benchmark-side index, so each training corpus is transferred and
scanned once rather than four times.

### Resumable pretraining scan

Do not use one repository-wide `load_dataset` stream for the multi-terabyte
pretraining mixes. Initialize a revision-pinned SQLite manifest once, then run
the resumable worker:

```bash
python scripts/manage_olmo_pretraining_scan.py \
  --manifest ../data/olmo_ground_truth/pretraining/olmo3_7b_manifest.sqlite \
  init \
  --repo allenai/dolma3_mix-6T-1025-7B \
  --revision 2ca900fbe14e86c5c83d064d9f0882f1c0b8c05b

python scripts/manage_olmo_pretraining_scan.py \
  --manifest ../data/olmo_ground_truth/pretraining/olmo3_7b_manifest.sqlite \
  run \
  --repo allenai/dolma3_mix-6T-1025-7B \
  --revision 2ca900fbe14e86c5c83d064d9f0882f1c0b8c05b \
  --output-dir ../data/olmo_ground_truth/pretraining/olmo3_7b_shards
```

`status` prints completed shard, byte, and document counts. `run` returns work
left in `running` by an interrupted process to `pending`; completed shards are
not rescanned. A file lock prevents two workers from accidentally reclaiming
each other's active shard. `--retry-failed` schedules each currently failed
shard for one additional attempt, and `--keep-going` lets the worker continue
after a failure.
Software-related source directories have queue priority, but all manifest
shards remain required for an exhaustive result. Each completed shard writes
only nonzero string evidence, atomically, while the manifest records negative
scan coverage and the document count. This avoids storing 1,597 negative rows
for every one of tens of thousands of shards.

After `status` reports that every shard is complete, `finalize --output PATH`
reduces the shard evidence to one best row per benchmark item. It refuses to
produce an apparently complete result while any shard is pending, running, or
failed.

For deterministic local validation, `--jsonl PATH` accepts JSONL rows and
recursively extracts string fields from plain-text or nested chat schemas. The
output keeps corpus, stage, document ID, scan count, exact flag, n-gram coverage,
threshold, and label in every benchmark-item row. The defaults (`n=13`, coverage
`0.8`) are provisional retrieval settings and must be frozen or recalibrated
without looking at Q1 results before a full scientific run.

## Not implemented in this first pass

- edit-distance and AST-based surface/program matching;
- embedding retrieval and semantic/paraphrase adjudication;
- TRACER's three LLM stages and validation against open-data evidence;
- an indexed search service that avoids transferring every pretraining shard;
- synthesis of separate method families into the final item-level label and
  measured proxy-label error rate *e*.

These omissions are explicit because exact/n-gram evidence alone is a lower
bound on contamination and cannot establish a clean label.
