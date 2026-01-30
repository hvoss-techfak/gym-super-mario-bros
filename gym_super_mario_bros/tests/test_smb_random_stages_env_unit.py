import numpy as np
import pytest

import gym_super_mario_bros.smb_random_stages_env as rse


class DummyStageEnv:
    def __init__(self, rom_mode="vanilla", target=None):
        self.rom_mode = rom_mode
        self.target = target
        self.closed = False
        self._screen = np.zeros((2, 2, 3), dtype=np.uint8)

    @property
    def screen(self):
        return self._screen

    def reset(self, seed=None, options=None, return_info=None):
        return ("obs", {"seed": seed, "target": self.target})

    def step(self, action):
        return ("obs", 0.0, False, False, {"action": action})

    def close(self):
        self.closed = True

    def get_keys_to_action(self):
        return {(): 0}

    def get_action_meanings(self):
        return ["NOOP"]


@pytest.fixture(autouse=True)
def _patch_underlying_env(monkeypatch):
    # Avoid creating 32 real NESEnv instances.
    monkeypatch.setattr(rse, "SuperMarioBrosEnv", DummyStageEnv)


def test_seed_none_returns_empty_list():
    env = rse.SuperMarioBrosRandomStagesEnv(rom_mode="vanilla")
    assert env.seed(None) == []


def test_seed_sets_rng_reproducibly():
    env = rse.SuperMarioBrosRandomStagesEnv(rom_mode="vanilla")
    env.seed(123)
    first = env.np_random.randint(0, 1000)
    env.seed(123)
    second = env.np_random.randint(0, 1000)
    assert first == second


def test_reset_selects_from_subset_stages(monkeypatch):
    env = rse.SuperMarioBrosRandomStagesEnv(rom_mode="vanilla", stages=["4-2"])
    obs, info = env.reset(seed=1)
    assert info["target"] == (4, 2)


def test_reset_can_override_stages_via_options():
    env = rse.SuperMarioBrosRandomStagesEnv(rom_mode="vanilla", stages=["1-1"])
    obs, info = env.reset(seed=1, options={"stages": ["8-4"]})
    assert info["target"] == (8, 4)


def test_step_delegates_to_current_env():
    env = rse.SuperMarioBrosRandomStagesEnv(rom_mode="vanilla", stages=["2-3"])
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(7)
    assert info["action"] == 7


def test_close_closes_all_child_envs_and_viewer():
    env = rse.SuperMarioBrosRandomStagesEnv(rom_mode="vanilla")

    class DummyViewer:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    env.viewer = DummyViewer()
    env.close()

    assert env.env is None
    assert env.viewer.closed is True

    # subsequent close should error
    with pytest.raises(ValueError):
        env.close()
