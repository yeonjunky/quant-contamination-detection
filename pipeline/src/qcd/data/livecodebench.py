"""LiveCodeBench loader (paper §4.2's primary contamination/control axis).

`datasets>=5` dropped script-based dataset loading (`datasets.load_dataset(
"livecodebench/code_generation_lite", ...)` now raises `RuntimeError: Dataset
scripts are no longer supported`, confirmed against the installed version
during pipeline construction) so this module bypasses the `datasets` library
entirely and downloads the raw per-release `test*.jsonl` files directly via
`huggingface_hub.hf_hub_download`.

`_RELEASE_FILES` mirrors the file manifest the (now-unusable) HF loading
script itself hardcodes as `ALLOWED_FILES` — copied here, not re-derived,
so a future release doesn't require guessing which files it adds. Verified
during pipeline construction that the six files are non-overlapping
`question_id` chunks in ascending `contest_date` order (test.jsonl starts
2023-05-07, matching CLAUDE.md §4.2's "LCB collection begins 2023-05"), so
concatenating a release's file list and never mixing releases is safe.

The split passed to this loader is the latest shared-control boundary. It
creates a collection envelope only; paper §4.2's per-arm labels are
materialized later in `data/temporal_labels.py`.
"""

from __future__ import annotations

import datetime as dt
import json

from qcd.data.schema import Dataset, Item

_REPO_ID = "livecodebench/code_generation_lite"
REPO_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"

# Mirrors ALLOWED_FILES from the dataset's (now dataset-scripts-unsupported)
# loading script, cross-checked during pipeline construction: each file's
# question_ids do not overlap with any other file's, and dates are ascending
# across files (test.jsonl: 2023-05-07..2024-03-02, ..., test6.jsonl:
# ..2025-04-06). Do not add a new release here without re-verifying that.
_RELEASE_FILES: dict[str, tuple[str, ...]] = {
    "release_v1": ("test.jsonl",),
    "release_v2": ("test.jsonl", "test2.jsonl"),
    "release_v3": ("test.jsonl", "test2.jsonl", "test3.jsonl"),
    "release_v4": ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"),
    "release_v5": ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"),
    "release_v6": ("test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"),
}

DEFAULT_RELEASE = "release_v6"


def _download_release_rows(release_version: str) -> list[dict]:
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    try:
        filenames = _RELEASE_FILES[release_version]
    except KeyError:
        raise ValueError(f"unknown LiveCodeBench release {release_version!r}; known: {sorted(_RELEASE_FILES)}") from None

    rows: list[dict] = []
    for filename in filenames:
        path = hf_hub_download(
            _REPO_ID, filename, repo_type="dataset", revision=REPO_REVISION,
        )
        with open(path, encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def _to_item(row: dict, release_version: str, contaminated: bool) -> Item:
    dataset = Dataset.LCB_PRE if contaminated else Dataset.LCB_POST
    # `metadata` (the row's own field, distinct from Item.metadata the dict
    # we're building) is a JSON string that carries `func_name` for
    # "functional" testtype problems (LeetCode-platform items) — required by
    # scoring/sandbox.py's functional harness, empty/absent for "stdin"
    # testtype problems (Codeforces/AtCoder-platform items).
    lcb_metadata = json.loads(row["metadata"]) if row.get("metadata") else {}
    return Item(
        item_id=row["question_id"],
        dataset=dataset,
        prompt=row["question_content"],
        difficulty=row.get("difficulty"),
        release_version=release_version,
        metadata={
            "platform": row.get("platform"),
            "contest_date": row["contest_date"],
            "contest_id": row.get("contest_id"),
            "starter_code": row.get("starter_code"),
            "func_name": lcb_metadata.get("func_name"),
            # Raw, undecoded field strings — public_test_cases is plain JSON;
            # private_test_cases is base64(zlib(pickle(json_string))), per
            # LiveCodeBench's own encoding (confirmed by direct decode during
            # pipeline construction). Decoded lazily in scoring/sandbox.py,
            # not here, so loading a split never pays that cost for items
            # that end up unscored.
            "public_test_cases": row.get("public_test_cases"),
            "private_test_cases": row.get("private_test_cases"),
        },
    )


def load_livecodebench_split(
    cutoff_boundary: dt.datetime,
    *,
    release_version: str = DEFAULT_RELEASE,
) -> tuple[list[Item], list[Item]]:
    """Load one release and split the collection envelope at the shared bound.

    `Dataset.LCB_PRE` means only "before the shared-control boundary" here;
    it is not a model-specific exposure label. `materialize_model_item_labels`
    later applies each model's own cutoff and excludes the intermediate
    `clean-by-model-cutoff` window from the primary contrast.
    """
    rows = _download_release_rows(release_version)
    pre: list[Item] = []
    post: list[Item] = []
    for row in rows:
        contest_date = dt.datetime.fromisoformat(row["contest_date"])
        contaminated = contest_date < cutoff_boundary
        item = _to_item(row, release_version, contaminated)
        (pre if contaminated else post).append(item)
    return pre, post
