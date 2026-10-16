#!/usr/bin/env python3
"""Print the resolved main-result configuration for one experiment."""
import argparse
import pprint

from cair.config import resolve_experiment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--il", choices=["bc", "dagger"], required=True)
    p.add_argument("--rl", choices=["sac", "ddpg", "ppo"], required=True)
    args = p.parse_args()
    pprint.pprint(resolve_experiment(args.task, args.il, args.rl), sort_dicts=False)


if __name__ == "__main__":
    main()
