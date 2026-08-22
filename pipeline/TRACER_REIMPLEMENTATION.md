# TRACER reimplementation specification

This document records the implementation boundary for paper section 5, step 5.
It was checked against arXiv:2605.24079, especially sections 3–5 and Appendix A.
TRACER has four pipeline stages, three of which call an LLM:

1. LLM instruction normalization;
2. embedding-based semantic triage;
3. LLM fine-grained verification;
4. LLM trivial-task screening.

This is a reimplementation, not the authors' code. The paper provides the three
LLM prompt templates, the embedding-model family, and the tuned thresholds, but
does not release executable code or the annotated task-pair dataset.

## Frozen paper-specified behavior

- Normalize benchmark and training-task descriptions independently.
- Embed normalized descriptions with `jina-embeddings-v3`.
- Route an embedding score `sigma >= 0.9` directly to Functionally Identical
  (`FI`).
- Route `sigma <= 0.6` directly to Unrelated (`U`).
- Send only `0.6 < sigma < 0.9` to LLM verification.
- Parse verification choices as `A=FI`, `B=NI`, `C=SL`, and `D=U`.
- For a contamination candidate, screen both descriptions independently for a
  trivial/basic-helper task. If either side is trivial, exclude the pair from
  contamination reporting.
- Preserve the pre-screen label and an explicit exclusion flag. Exclusion must
  not be silently rewritten as `U`.

The `> 0.5` embedding cutoff in paper section 5.1 was used only to construct the
manual evaluation candidate pool. It is not a TRACER inference threshold.

## Reimplementation choices that remain to be frozen

The paper does not specify the following operational values:

- exact Jina checkpoint revision and embedding task/instruction;
- similarity function, pooling, truncation, and maximum input length;
- exact LLM checkpoint or API snapshot;
- system prompt, temperature, top-p, seed, and output-token limit;
- invalid-output retry policy;
- normalization caching and duplicate-description handling;
- whether trivial screening is called for pairs already labeled `U`.

Each value must be recorded in the run manifest as a reimplementation choice.
None may be described as an original-paper setting. The default implementation
will avoid screening `U` pairs because they are not contamination candidates,
while retaining enough state to audit that routing decision.

The provisional local adapter pins `jinaai/jina-embeddings-v3` by an immutable
Hub commit, uses its symmetric `text-matching` task and explicit L2-normalized
cosine, and clips negative cosine values to zero. This last transformation makes
the score conform to the paper's stated `[0, 1]` domain and cannot alter routing
under the paper's lower threshold of `0.6`. These are reimplementation choices,
not settings reported by TRACER.

## Candidate retrieval boundary

The original experiments compare roughly two million benchmark–training pairs.
Olmo pretraining contains billions of documents, so running Jina over the full
Cartesian product is not operationally feasible. Candidate retrieval therefore
precedes TRACER:

1. exact/token n-gram and later BM25 retrieval scan the revision-pinned corpus;
2. candidate task pairs retain source shard, document ID, retrieval method, and
   retrieval score;
3. TRACER runs only on those candidate pairs;
4. a no-candidate item records completed corpus coverage rather than being
   silently treated as a clean ground-truth label.

String-match negatives are not clean labels. They establish only that no
surface-form evidence was found at the frozen retrieval setting.

## Required pair-level record

Every routed candidate must retain:

- benchmark, item ID, corpus, revision, shard, and document ID;
- original and normalized descriptions on both sides;
- retrieval method and score;
- embedding model, revision, score, and triage route;
- raw verification output and parsed `FI/NI/SL/U` label;
- raw trivial-screen outputs and parsed decisions for both sides;
- pre-screen label, exclusion flag, final fine-grained label, and nullable
  binary contamination label;
- normalizer, verifier, and screener model identities;
- prompt version, decoding configuration, and run timestamp.

Excluded pairs receive a null binary label. `FI`, `NI`, and `SL` map to true;
`U` maps to false.

## Validation before Q1b

- Unit-test exact routing at `0.6` and `0.9`.
- Strictly reject malformed, multiple-choice, or missing-label LLM outputs.
- Verify that either-side triviality produces exclusion without destroying the
  original semantic label.
- Measure recall against Olmo exact/n-gram positives.
- Because string negatives are not verified clean examples, estimate
  specificity only on a manually adjudicated stratified sample.
- Audit normalization meaning preservation and the paper's principal failure
  modes: core-logic errors, adjacent-category confusion, hallucinated task
  properties, and normalization distortion.

TRACER-derived labels remain unavailable for Q1b until these validation steps
and the pretraining candidate search are complete.
