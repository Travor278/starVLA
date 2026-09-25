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
    run = runs[0] if len(runs) == 1 else {}
    schedule = {
        int(row["episode_id"]): row for row in run.get("schedule", [])
    }
    schedule_fields = (
        "condition", "layout_id", "reset_seed", "generalization_seed",
        "generalization_sample_index",
    )
    schedule_matches = len(schedule) == 100 and all(
        int(record["episode_id"]) in schedule
        and all(
            record.get(field) == schedule[int(record["episode_id"])].get(field)
            for field in schedule_fields
        )
        for record in episodes
    )
    layout_counts = Counter(str(row["layout_id"]) for row in episodes)
    layout_balanced = sorted(layout_counts.values()) == [33, 33, 34]
    seed_balanced = all(
        Counter(
            int(row["generalization_seed"])
            for row in episodes if row["condition"] == condition
        ) == {seed: 4 for seed in range(1001, 1006)}
        for condition in CONDITIONS - {"nominal"}
    )
    complete = (
        len(runs) == 1
        and run.get("standard_protocol") is True
        and len(episodes) == 100
        and set(ids) == set(range(100))
        and counts == {condition: 20 for condition in CONDITIONS}
        and schedule_matches
        and layout_balanced
        and seed_balanced
        and len(summaries) == 1
        and summaries[0]["overall"]["episodes"] == 100
        and not any(row.get("event") == "error" for row in rows)
    )
    return {
        "complete_standard_evaluation": complete,
        "task_id": runs[0].get("task_id") if runs else None,
        "episode_count": len(episodes),
        "condition_counts": dict(sorted(counts.items())),
        "layout_counts": dict(sorted(layout_counts.items())),
        "schedule_matches": schedule_matches,
        "generalization_seeds_balanced": seed_balanced,
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
