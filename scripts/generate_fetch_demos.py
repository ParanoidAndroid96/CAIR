#!/usr/bin/env python3
"""Generate Fetch demonstrations using the supplied rule-based controllers.

This is intentionally independent of the historical replay-buffer class.  It
writes ordinary NumPy arrays that can be consumed by any learner.
"""
import argparse
from pathlib import Path

import numpy as np

from cair.demos.fetch import make_fetch_expert

ENV_IDS = {
    "push": "FetchPush-v1",
    "slide": "FetchSlide-v1",
    "pick_and_place": "FetchPickAndPlace-v1",
}


def _make_env(env_id):
    # The historical runs used legacy Gym IDs. Try Gymnasium-Robotics first,
    # then legacy Gym for old environments.
    try:
        import gymnasium as gym
        try:
            import gymnasium_robotics  # noqa: F401
            try:
                gym.register_envs(gymnasium_robotics)
            except AttributeError:
                pass
        except ImportError:
            pass
        return gym.make(env_id)
    except Exception as first_error:
        try:
            import gym
            return gym.make(env_id)
        except Exception as second_error:
            raise RuntimeError(
                "Could not create {!r} with Gymnasium or legacy Gym. "
                "Use --env-id for the equivalent Fetch environment installed "
                "on your system. Gymnasium-Robotics renamed some legacy IDs.\n"
                "Gymnasium error: {}\nGym error: {}".format(env_id, first_error, second_error)
            )


def _reset(env, seed):
    out = env.reset(seed=seed) if seed is not None else env.reset()
    return out[0] if isinstance(out, tuple) else out


def _step(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
    else:
        obs, reward, done, info = out
    return obs, float(reward), bool(done), info


def flatten_obs(obs):
    # Historical Fetch replay buffers used the observation vector. Preserve
    # goals as separate arrays as well so the generated file is self-contained.
    return np.asarray(obs["observation"], dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=sorted(ENV_IDS), required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env-id", default=None, help="Override the paper's legacy Gym environment ID.")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    env_id = args.env_id or ENV_IDS[args.task]
    env = _make_env(env_id)
    expert = make_fetch_expert(args.task)

    states, actions, rewards, dones, next_states = [], [], [], [], []
    desired_goals, next_desired_goals = [], []

    for ep in range(args.episodes):
        obs = _reset(env, args.seed + ep)
        expert.reset()
        while True:
            action = np.asarray(expert.act(obs), dtype=np.float32)
            next_obs, reward, done, _ = _step(env, action)
            states.append(flatten_obs(obs))
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            next_states.append(flatten_obs(next_obs))
            desired_goals.append(np.asarray(obs["desired_goal"], dtype=np.float32))
            next_desired_goals.append(np.asarray(next_obs["desired_goal"], dtype=np.float32))
            obs = next_obs
            if done:
                break

    env.close()
    output = Path(args.output or "data/fetch/{}_50traj.npz".format(args.task))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        state=np.asarray(states),
        action=np.asarray(actions),
        reward=np.asarray(rewards, dtype=np.float32),
        done=np.asarray(dones, dtype=np.bool_),
        next_state=np.asarray(next_states),
        desired_goal=np.asarray(desired_goals),
        next_desired_goal=np.asarray(next_desired_goals),
    )
    print("Saved {} transitions from {} trajectories to {}".format(len(states), args.episodes, output))


if __name__ == "__main__":
    main()
