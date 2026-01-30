import numpy as np
import pytest

from gym_super_mario_bros._registration import make


def test_checkpoint_restore_is_deterministic_for_future_steps():
    """After restoring a checkpoint, the env should produce identical rollouts.

    We compare observations/rewards/termination flags/selected info fields for a
    fixed action sequence.
    """
    env = make("SuperMarioBros-v0")

    obs0, info0 = env.reset(seed=0)

    # Take a few steps to get away from power-on state.
    warmup_actions = [0, 1, 0, 1, 1, 0, 0]
    for a in warmup_actions:
        env.step(a)

    # Save checkpoint at an arbitrary point.
    base = env.unwrapped
    ckpt = base.save_checkpoint()

    # Generate a deterministic action sequence to compare.
    actions = [0, 1, 2, 3, 4, 5, 6, 7] * 3

    rollout_1 = []
    for a in actions:
        obs, reward, terminated, truncated, info = env.step(a)
        rollout_1.append((obs.copy(), float(reward), bool(terminated), bool(truncated), dict(info)))
        if terminated or truncated:
            break

    # Restore and roll again.
    base.load_checkpoint(ckpt)

    rollout_2 = []
    for a in actions:
        obs, reward, terminated, truncated, info = env.step(a)
        rollout_2.append((obs.copy(), float(reward), bool(terminated), bool(truncated), dict(info)))
        if terminated or truncated:
            break

    assert len(rollout_1) == len(rollout_2)

    for (o1, r1, t1, tr1, i1), (o2, r2, t2, tr2, i2) in zip(rollout_1, rollout_2):
        assert np.array_equal(o1, o2)
        assert r1 == r2
        assert t1 == t2
        assert tr1 == tr2
        for k in ["coins", "flag_get", "life", "world", "score", "stage", "time", "x_pos", "y_pos"]:
            assert i1.get(k) == i2.get(k)

    env.close()


def test_checkpoint_restore_rolls_back_ram_changes():
    env = make("SuperMarioBros-v0")
    env.reset(seed=0)

    base = env.unwrapped

    # Save.
    ckpt = base.save_checkpoint()

    # Make a visible RAM change.
    addr = 0x075A  # life counter
    before = int(base.ram[addr])
    base.ram[addr] = 0
    assert int(base.ram[addr]) == 0

    # Restore.
    base.load_checkpoint(ckpt)
    assert int(base.ram[addr]) == before

    env.close()
