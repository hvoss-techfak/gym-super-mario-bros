import numpy as np


def test_step_sets_terminated_when_game_over(monkeypatch):
    """Even if nes-py doesn't report done that frame, env should terminate on game over."""
    from gym_super_mario_bros.smb_env import SuperMarioBrosEnv

    # Create an instance without running full emulator init.
    env = object.__new__(SuperMarioBrosEnv)

    # Minimal RAM backing for _life (0x075A) and other reads.
    env.ram = np.zeros(2048, dtype=np.uint8)
    env.ram[0x075A] = 0xFF  # game over

    # Not a single-stage env => _get_done should use _is_game_over
    env._target_world, env._target_stage, env._target_area = None, None, None

    # Stub NESEnv.step to return legacy 4-tuple with done=False.
    def fake_super_step(_self, action):
        obs = np.zeros((240, 256, 3), dtype=np.uint8)
        reward = 0.0
        done = False
        info = {}
        return obs, reward, done, info

    monkeypatch.setattr(
        SuperMarioBrosEnv.__mro__[1],  # NESEnv
        "step",
        fake_super_step,
        raising=True,
    )

    obs, reward, terminated, truncated, info = SuperMarioBrosEnv.step(env, 0)
    assert terminated is True
    assert truncated is False


def test_step_preserves_truncated(monkeypatch):
    """If underlying env reports truncated, we keep it (and may also terminate)."""
    from gym_super_mario_bros.smb_env import SuperMarioBrosEnv

    env = object.__new__(SuperMarioBrosEnv)
    env.ram = np.zeros(2048, dtype=np.uint8)
    env.ram[0x075A] = 0x00  # not game over
    env._target_world, env._target_stage, env._target_area = None, None, None

    def fake_super_step(_self, action):
        obs = np.zeros((240, 256, 3), dtype=np.uint8)
        reward = 0.0
        terminated = False
        truncated = True
        info = {}
        return obs, reward, terminated, truncated, info

    monkeypatch.setattr(
        SuperMarioBrosEnv.__mro__[1],
        "step",
        fake_super_step,
        raising=True,
    )

    _obs, _reward, terminated, truncated, _info = SuperMarioBrosEnv.step(env, 0)
    assert terminated is False
    assert truncated is True
