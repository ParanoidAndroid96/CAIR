"""Rule-based Fetch demonstrators used in the original CAIR experiments.

The controllers are refactored from the supplied experimental environment
wrappers.  They operate directly on the standard Fetch observation dictionary,
so the same controller can be used to generate offline trajectories or queried
online for DAgger.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np


class _BaseFetchExpert:
    def __init__(self) -> None:
        self.mode = 0
        self.total_steps = 0
        self.mode_freq = [0.0, 0.0, 0.0, 0.0]

    def reset(self) -> None:
        self.mode = 0
        self.total_steps = 0
        self.mode_freq = [0.0, 0.0, 0.0, 0.0]

    @staticmethod
    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    @staticmethod
    def _angle(source_x: float, source_y: float, target_x: float, target_y: float) -> float:
        dx = target_x - source_x
        dy = target_y - source_y
        if dx == 0.0 and dy != 0.0:
            return 90.0 if dy > 0 else 270.0
        if dx == 0.0 and dy == 0.0:
            return 0.0
        angle = math.atan(dy / dx) * 180.0 / math.pi
        if target_x < source_x:
            angle = 180.0 + angle
        elif target_y < source_y and target_x > source_x:
            angle = 360.0 + angle
        return angle

    def _unpack(self, obs: Dict[str, np.ndarray]):
        raw = np.asarray(obs["observation"], dtype=np.float64)
        goal = np.asarray(obs["desired_goal"], dtype=np.float64)
        object_pos = raw[3:6]
        gripper_pos = raw[:3]
        gripper_state = raw[9:11]
        object_rel_pos = raw[6:9]
        return goal, object_pos, gripper_pos, gripper_state, object_rel_pos


class FetchPushExpert(_BaseFetchExpert):
    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        goal, object_pos, gripper_pos, _, object_rel = self._unpack(obs)
        self.mode_freq[self.mode] += 1

        oriented = object_rel.copy()
        oriented[2] += 0.03

        opp_goal_ang = (self._angle(object_pos[0], object_pos[1], goal[0], goal[1]) + 180.0) * math.pi / 180.0
        x_disp = 0.08 * math.cos(opp_goal_ang)
        y_disp = 0.08 * math.sin(opp_goal_ang)
        proximity = object_pos.copy()
        proximity[0] = min(proximity[0] + 3.0 * x_disp, 1.57)
        proximity[1] += 3.0 * y_disp
        proximity[2] = 0.52

        proximity_dist = self._dist(gripper_pos, proximity)
        obj_dist = self._dist(gripper_pos, object_pos)
        action = np.zeros(4, dtype=np.float32)

        if proximity_dist > 0.1:
            delta = proximity - gripper_pos
            action[:3] = delta * np.asarray([10.0, 10.0, 20.0])
        else:
            self.mode = 1

        if self.mode >= 1:
            if obj_dist > 0.06 and self.mode == 1:
                action[:3] = (object_pos - gripper_pos) * 10.0
                action[2] = -1.0
                action[3] = 0.0
            else:
                action[:3] = (goal - object_pos) * 3.0
                action[3] = 0.0
                self.mode = 2

        self.total_steps += 1
        return action


class FetchSlideExpert(_BaseFetchExpert):
    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        goal, object_pos, gripper_pos, _, object_rel = self._unpack(obs)
        self.mode_freq[self.mode] += 1

        oriented = object_rel.copy()
        oriented[2] += 0.03
        opp_goal_ang = (self._angle(object_pos[0], object_pos[1], goal[0], goal[1]) + 180.0) * math.pi / 180.0
        proximity = object_pos.copy()
        proximity[0] += 0.2 * math.cos(opp_goal_ang)
        proximity[1] += 0.2 * math.sin(opp_goal_ang)
        proximity[2] = 0.52

        proximity_dist = self._dist(gripper_pos, proximity)
        obj_dist = self._dist(gripper_pos, object_pos)
        goal_dist = self._dist(gripper_pos, goal)
        obj_goal_dist = self._dist(object_pos, goal)
        action = np.zeros(4, dtype=np.float32)

        if obj_dist > 0.25 or goal_dist < obj_goal_dist + 0.04:
            delta = proximity - gripper_pos
            action[0] = 10.0 * delta[0]
            action[1] = 10.0 * delta[1]
            action[2] = 0.0 if goal_dist > obj_goal_dist + 0.04 else 20.0 * delta[2]
        elif self.mode != 2:
            self.mode = 1

        if self.mode >= 1:
            proximity2 = proximity.copy()
            proximity2[2] = object_pos[2]
            proximity_dist2 = self._dist(gripper_pos, proximity2)
            if proximity_dist2 > 0.04 and obj_dist > 0.04 and self.mode == 1:
                action[:3] = (proximity2 - gripper_pos) * 10.0
                action[3] = 0.0
            else:
                action[:3] = (goal - gripper_pos) * 1.75
                action[2] = 0.0
                action[3] = 0.0
                self.mode = 2

        self.total_steps += 1
        return action


class FetchPickAndPlaceExpert(_BaseFetchExpert):
    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        goal, object_pos, gripper_pos, gripper_state, object_rel = self._unpack(obs)
        self.mode_freq[self.mode] += 1
        oriented = object_rel.copy()
        oriented[2] += 0.03

        obj_dist = self._dist(gripper_pos, object_pos)
        action = np.zeros(4, dtype=np.float32)
        if obj_dist > 0.03:
            action[:3] = oriented * 6.0
            action[3] = 0.3
            self.mode = 0
        else:
            if gripper_state[0] >= 0.03:
                action[:3] = object_rel * 6.0
                self.mode = 1
            else:
                action[:3] = (goal - object_pos) * 6.0
                self.mode = 2
            action[3] = -0.3

        self.total_steps += 1
        return action


def make_fetch_expert(name: str) -> _BaseFetchExpert:
    key = name.lower().replace("-", "_")
    if key in {"push", "fetch_push"}:
        return FetchPushExpert()
    if key in {"slide", "fetch_slide"}:
        return FetchSlideExpert()
    if key in {"pick_and_place", "fetch_pick_and_place", "pnp"}:
        return FetchPickAndPlaceExpert()
    raise ValueError("Unknown Fetch controller: {}".format(name))
