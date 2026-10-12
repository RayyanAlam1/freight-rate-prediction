"""Single entry point: train, predict, then validate both files with the provided scorer.

Run:  python run.py
"""
from __future__ import annotations

import subprocess
import sys

import predict
import train
from freight_rate.data import DECEMBER_FILE, PREDICTIONS_FILE, ROOT


def main() -> None:
    print("== 1/3 train ==")
    train.main()
    print("\n== 2/3 predict ==")
    predict.main()
    print("\n== 3/3 provided scorer ==")
    sys.stdout.flush()  # keep our output ahead of the scorer's when piped to a file
    subprocess.run(
        [sys.executable, "score.py", "--predictions", PREDICTIONS_FILE.name,
         "--december-predictions", DECEMBER_FILE.relative_to(ROOT).as_posix()],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
