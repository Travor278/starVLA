"""Register StarVLA, then delegate to Evo-DexBench's fixed protocol CLI."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def main() -> None:
    benchmark_root = os.environ.get("EVODEX_ROOT")
    if not benchmark_root:
        raise RuntimeError("Set EVODEX_ROOT to the pinned Evo-DexBench checkout")
    benchmark_path = Path(benchmark_root).expanduser().resolve(strict=True)
    sys.path.insert(0, str(benchmark_path))

    from dex_benchmark.policies import PolicyRegistry
    from examples.simBenchmarks.EvoDexBench.eval_files.bridge import StarVLAPolicySession
    from scripts.policies.evaluate_policy_protocol import main as evaluate_main

    PolicyRegistry.register("starvla", StarVLAPolicySession)
    evaluate_main()


if __name__ == "__main__":
    main()
