"""Check whether an Evo-DexBench JSONL log is a complete standard evaluation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

CONDITIONS = {"nominal", "spatial", "camera", "lighting", "physical"}


def verify(path: Path) -> dict[str, object]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    runs = [row for row in rows if row.get("event") == "run"]
    episodes = [row["record"] for row in rows if row.get("event") == "episode"]
    summaries = [row["summary"] for row in rows if row.get("event") == "summary"]
    counts = Counter(str(row["condition"]) for row in episodes)
    ids = [int(row["episode_id"]) for row in episodes]
    complete = (
        len(runs) == 1
        and runs[0].get("standard_protocol") is True
        and len(episodes) == 100
        and set(ids) == set(range(100))
        and counts == {condition: 20 for condition in CONDITIONS}
        and len(summaries) == 1
        and summaries[0]["overall"]["episodes"] == 100
        and not any(row.get("event") == "error" for row in rows)
    )
    return {
        "complete_standard_evaluation": complete,
        "task_id": runs[0].get("task_id") if runs else None,
        "episode_count": len(episodes),
        "condition_counts": dict(sorted(counts.items())),
        "summary_present": len(summaries) == 1,
        "log": str(path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    result = verify(args.log)
    print(json.dumps(result, indent=2))
    if not result["complete_standard_evaluation"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
