from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from dataset import load_cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None, help="default cases-root/_cache")
    parser.add_argument("--refresh", action="store_true", help="delete and rebuild the cache")
    parser.add_argument("--no-normalize", action="store_true", help="skip percentile-clip + z-score")
    args = parser.parse_args()

    cache_dir = args.cache_dir or (args.cases_root / "_cache")
    if args.refresh:
        shutil.rmtree(cache_dir, ignore_errors=True)

    cases = load_cases(args.cases_root, cache_dir, normalize=not args.no_normalize)
    print(f"cached {len(cases)} cases -> {cache_dir}", flush=True)


if __name__ == "__main__":
    main()
