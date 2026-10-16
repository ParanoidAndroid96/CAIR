#!/usr/bin/env python3
"""Download one or all locomotion demonstration datasets used by CAIR."""
import argparse

from cair.demos.minari import DATASETS, load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=sorted(DATASETS) + ["all"], default="all")
    args = p.parse_args()
    tasks = sorted(DATASETS) if args.task == "all" else [args.task]
    for task in tasks:
        print("Loading {} -> {}".format(task, DATASETS[task]))
        ds = load_dataset(task, download=True)
        print("  ready:", ds)


if __name__ == "__main__":
    main()
