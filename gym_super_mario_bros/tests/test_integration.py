import os

import pytest

from gym_super_mario_bros._registration import make


def test_make_and_step_smoke():
    env = make("SuperMarioBros-v0")
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(0)

    assert truncated is False
    assert isinstance(info, dict)
    for k in ["coins", "flag_get", "life", "world", "score", "stage", "time", "x_pos"]:
        assert k in info

    env.close()
