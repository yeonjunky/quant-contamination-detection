"""Which training corpus belongs to which Olmo checkpoint.

Paper §5, step 5 searches "the released Olmo3 pretraining and post-training
corpora". The design has *two* Olmo arms (`models/registry.py`:
`Olmo3-7B-Instruct` and `Olmo3.1-32B-Instruct`), and they do not share a bulk
pretraining mix — `OLMO_GROUND_TRUTH.md`'s verified-inputs table records
`allenai/dolma3_mix-6T-1025-7B` for the 7B and `allenai/dolma3_mix-6T` for the
32B. A corpus-reference result therefore belongs to one checkpoint, not to
"the Olmo3 corpus": a `confirmed-match` in the 7B mix says nothing about the
32B arm, and a `no-match-found` in one mix is not a `no-match-found` in the
other.

Nothing here is inferred. Every repository, revision and model assignment is
either copied from `OLMO_GROUND_TRUTH.md`'s verified-inputs table or left
explicitly unresolved:

- the 32B pretraining mix has no pinned scan revision recorded anywhere in
  this repository, so `revision` is `None` and `require_revision` refuses to
  start a scan until an operator pins one;
- the three `Dolci-Instruct-*` post-training repositories are recorded without
  a per-checkpoint assignment. Whether the 7B and 32B Instruct checkpoints are
  post-trained on the same mixes is not established by anything this
  repository has verified, so they are listed as unassigned rather than
  silently attributed to both arms.

Closed-corpus arms (Qwen2.5, Llama-3.1) are `not-observable` on paper §4.2's
corpus axis by construction — there is no released corpus to search.
"""

from __future__ import annotations

import dataclasses

from qcd.data.schema import CorpusReferenceStatus

#: Registry names (`models/registry.py`) of the two open-corpus arms.
OPEN_CORPUS_MODELS: tuple[str, ...] = ("Olmo3-7B-Instruct", "Olmo3.1-32B-Instruct")

#: Registry names whose training corpora are not released (paper §4.5.2).
CLOSED_CORPUS_MODELS: tuple[str, ...] = (
    "Qwen2.5-7B-Instruct", "Qwen2.5-32B-Instruct", "Llama-3.1-8B-Instruct",
)

#: The stages `scripts/search_olmo_corpus.py` accepts, in training order.
STAGES: tuple[str, ...] = ("pretraining", "sft", "dpo", "rlvr")


@dataclasses.dataclass(frozen=True)
class OlmoCorpus:
    """One released corpus repository, with whatever is actually known about it.

    `models` is `None` when this repository's assignment to a specific
    checkpoint has not been verified here — not a shorthand for "all arms".
    """

    stage: str
    hf_repo: str
    revision: str | None
    models: tuple[str, ...] | None
    note: str

    @property
    def assignment_verified(self) -> bool:
        return self.models is not None

    def corpus_id(self) -> str:
        """`repo@revision` — the `corpus` field written into evidence rows."""
        return f"{self.hf_repo}@{self.revision}" if self.revision else self.hf_repo


#: Copied from `OLMO_GROUND_TRUTH.md`'s "Verified public inputs (2026-08-21)"
#: table. Do not add an entry without a source for the repository *and* for the
#: checkpoint it belongs to.
OLMO_CORPORA: tuple[OlmoCorpus, ...] = (
    OlmoCorpus(
        stage="pretraining",
        hf_repo="allenai/dolma3_mix-6T-1025-7B",
        revision="2ca900fbe14e86c5c83d064d9f0882f1c0b8c05b",
        models=("Olmo3-7B-Instruct",),
        note=(
            "7B bulk pretraining mix; scan revision pinned in OLMO_GROUND_TRUTH.md. "
            "The 7B card warns that some olmOCR science-PDF texts were redacted after "
            "training, so a negative search is not proof of non-exposure."
        ),
    ),
    OlmoCorpus(
        stage="pretraining",
        hf_repo="allenai/dolma3_mix-6T",
        revision=None,
        models=("Olmo3.1-32B-Instruct",),
        note=(
            "32B bulk pretraining mix. No scan revision has been pinned for it in this "
            "repository; pin one and record it before a scientific scan."
        ),
    ),
    OlmoCorpus(
        stage="sft",
        hf_repo="allenai/Dolci-Instruct-SFT",
        revision=None,
        models=None,
        note="Post-training SFT mix; per-checkpoint assignment not verified here.",
    ),
    OlmoCorpus(
        stage="dpo",
        hf_repo="allenai/Dolci-Instruct-DPO",
        revision=None,
        models=None,
        note="Post-training DPO mix; per-checkpoint assignment not verified here.",
    ),
    OlmoCorpus(
        stage="rlvr",
        hf_repo="allenai/Dolci-Instruct-RL",
        revision=None,
        models=None,
        note="Post-training RL/RLVR prompt mixture; per-checkpoint assignment not verified here.",
    ),
)


def corpora_for_model(model: str) -> tuple[OlmoCorpus, ...]:
    """Corpora verifiably belonging to `model`. Empty for a closed-corpus arm."""
    return tuple(
        corpus for corpus in OLMO_CORPORA
        if corpus.models is not None and model in corpus.models
    )


def unassigned_corpora() -> tuple[OlmoCorpus, ...]:
    """Released corpora whose owning checkpoint(s) this repository has not
    verified. Scanning one is allowed; attributing the result to a specific arm
    without a source is not."""
    return tuple(corpus for corpus in OLMO_CORPORA if corpus.models is None)


def find_corpus(hf_repo: str) -> OlmoCorpus | None:
    for corpus in OLMO_CORPORA:
        if corpus.hf_repo == hf_repo:
            return corpus
    return None


def require_revision(corpus: OlmoCorpus) -> str:
    if not corpus.revision:
        raise ValueError(
            f"{corpus.hf_repo} has no pinned scan revision in olmo_corpora.py "
            f"({corpus.note}). Pass an explicit --revision and record it before "
            "treating the scan as a scientific result."
        )
    return corpus.revision


def check_model_corpus(model: str, hf_repo: str) -> str | None:
    """Reject a (checkpoint, repository) pair the registry contradicts.

    Returns a warning string when the pairing cannot be checked (an unknown
    repository, or one whose checkpoint assignment is unverified), and None
    when the pairing is a recorded one.
    """
    if model in CLOSED_CORPUS_MODELS:
        raise ValueError(
            f"{model} has no released training corpus (paper §4.5.2); its corpus-reference "
            f"status is '{CorpusReferenceStatus.NOT_OBSERVABLE.value}' and no scan applies."
        )
    if model not in OPEN_CORPUS_MODELS:
        raise ValueError(
            f"unknown model {model!r}; open-corpus arms are {list(OPEN_CORPUS_MODELS)}"
        )
    corpus = find_corpus(hf_repo)
    if corpus is None:
        return (
            f"{hf_repo} is not in olmo_corpora.py; the attribution of this scan to {model} "
            "is operator-declared and unverified."
        )
    if corpus.models is None:
        return (
            f"{hf_repo} ({corpus.stage}) has no verified per-checkpoint assignment "
            f"({corpus.note}); attributing this scan to {model} is operator-declared."
        )
    if model not in corpus.models:
        raise ValueError(
            f"{hf_repo} is recorded as the corpus of {list(corpus.models)}, not {model}. "
            "The two Olmo arms do not share a bulk pretraining mix "
            "(OLMO_GROUND_TRUTH.md); scan each checkpoint's own corpus."
        )
    return None


def default_status(model: str) -> CorpusReferenceStatus | None:
    """Paper §4.2's corpus-axis value a closed-corpus arm carries without any
    search. `None` for an open-corpus arm, whose status comes from a scan."""
    if model in CLOSED_CORPUS_MODELS:
        return CorpusReferenceStatus.NOT_OBSERVABLE
    return None
