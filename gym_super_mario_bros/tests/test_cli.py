import builtins
import sys

import pytest

import gym_super_mario_bros._app.cli as cli


def test_get_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"])
    args = cli._get_args()
    assert args.env == "SuperMarioBros-v0"
    assert args.mode == "human"
    assert args.actionspace == "nes"
    assert args.steps == 500
    assert args.stages is None


def test_main_rejects_stages_for_non_random_env(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["prog", "--env", "SuperMarioBros-v0", "--stages", "1-1"])

    # avoid really exiting the test runner
    def _fake_exit(code=0):
        raise SystemExit(code)

    monkeypatch.setattr(sys, "exit", _fake_exit)

    with pytest.raises(SystemExit) as e:
        cli.main()

    assert e.value.code == 1
    out = capsys.readouterr().out
    assert "--stages" in out


def test_main_human_mode_calls_play_human(monkeypatch):
    calls = {"make": [], "play_human": 0}

    class DummyEnv:
        pass

    def fake_make(env_id, stages=None):
        calls["make"].append((env_id, stages))
        return DummyEnv()

    def fake_play_human(env):
        assert isinstance(env, DummyEnv)
        calls["play_human"] += 1

    monkeypatch.setattr(cli.gym, "make", fake_make)
    monkeypatch.setattr(cli, "play_human", fake_play_human)
    monkeypatch.setattr(cli, "play_random", lambda env, steps: (_ for _ in ()).throw(AssertionError("play_random should not be called")))

    monkeypatch.setattr(sys, "argv", ["prog", "--env", "SuperMarioBros-v0", "--mode", "human"])
    cli.main()

    assert calls["make"] == [("SuperMarioBros-v0", None)]
    assert calls["play_human"] == 1


def test_main_random_mode_calls_play_random_with_steps(monkeypatch):
    calls = {"play_random": []}

    class DummyEnv:
        pass

    monkeypatch.setattr(cli.gym, "make", lambda env_id, stages=None: DummyEnv())
    monkeypatch.setattr(cli, "play_human", lambda env: (_ for _ in ()).throw(AssertionError("play_human should not be called")))

    def fake_play_random(env, steps):
        assert isinstance(env, DummyEnv)
        calls["play_random"].append(steps)

    monkeypatch.setattr(cli, "play_random", fake_play_random)

    monkeypatch.setattr(sys, "argv", ["prog", "--env", "SuperMarioBros-v0", "--mode", "random", "--steps", "12"])
    cli.main()

    assert calls["play_random"] == [12]


def test_main_wraps_actionspace(monkeypatch):
    class DummyEnv:
        pass

    wrapped = {"called": False, "actions": None}

    def fake_make(env_id, stages=None):
        return DummyEnv()

    def fake_joypad_space(env, actions):
        assert isinstance(env, DummyEnv)
        wrapped["called"] = True
        wrapped["actions"] = actions
        return env

    monkeypatch.setattr(cli.gym, "make", fake_make)
    monkeypatch.setattr(cli, "JoypadSpace", fake_joypad_space)
    monkeypatch.setattr(cli, "play_human", lambda env: None)

    monkeypatch.setattr(sys, "argv", ["prog", "--actionspace", "right"])
    cli.main()

    assert wrapped["called"] is True
    assert wrapped["actions"] == cli._ACTION_SPACES["right"]
