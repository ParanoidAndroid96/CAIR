#!/usr/bin/env python3
"""Check external demonstration/expert resources without loading huge files."""
import argparse
from pathlib import Path

from cair.demos.minari import DATASETS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data")
    args = p.parse_args()
    root = Path(args.data_root)

    expected = {
        "Hammer BC demos": root / "adroit" / "hammer-demos-v0-bc-combined.hdf5",
        "Hammer interactive expert": root / "adroit" / "expert.zip",
        "Pen BC demos": root / "adroit" / "pen-v0_demos.pickle",
        "Pen interactive expert": root / "adroit" / "pen-v0.pickle",
    }
    print("Adroit resources:")
    for name, path in expected.items():
        print("  {:28s} {} {}".format(name, "OK" if path.exists() else "MISSING", path))

    print("\nLocomotion Minari IDs (download on demand):")
    for task, dataset_id in sorted(DATASETS.items()):
        print("  {:12s} {}".format(task, dataset_id))


if __name__ == "__main__":
    main()
