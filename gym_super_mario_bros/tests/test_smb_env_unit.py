import numpy as np
import pytest

from gym_super_mario_bros.smb_env import SuperMarioBrosEnv


def _make_env_without_init() -> SuperMarioBrosEnv:
    # Construct instance without running the heavy NESEnv.__init__.
    env = object.__new__(SuperMarioBrosEnv)
    env.ram = np.zeros(0x0800, dtype=np.uint8)
    env._target_world = None
    env._target_stage = None
    env._target_area = None
    env._time_last = 0
    env._x_position_last = 0
    return env


def test_read_mem_range_decodes_decimal_digits():
    env = _make_env_without_init()
    env.ram[0x10:0x13] = [1, 2, 3]
    assert env._read_mem_range(0x10, 3) == 123


def test_world_stage_area_score_time_coins_life_properties():
    env = _make_env_without_init()

    env.ram[0x075f] = 0  # world register -> world 1
    env.ram[0x075c] = 2  # stage register -> stage 3
    env.ram[0x0760] = 4  # area register -> area 5

    env.ram[0x07de:0x07de + 6] = [0, 0, 0, 1, 2, 3]
    env.ram[0x07f8:0x07f8 + 3] = [4, 0, 0]
    env.ram[0x07ed:0x07ed + 2] = [9, 9]
    env.ram[0x075a] = 2

    assert env._world == 1
    assert env._stage == 3
    assert env._area == 5
    assert env._score == 123
    assert env._time == 400
    assert env._coins == 99
    assert env._life == 2


def test_x_position_uses_page_and_offset_as_ints():
    env = _make_env_without_init()
    env.ram[0x6D] = 1
    env.ram[0x86] = 5
    assert env._x_position == 0x100 + 5


def test_left_x_position_wraps_u8():
    env = _make_env_without_init()
    env.ram[0x86] = 1
    env.ram[0x071C] = 250
    assert int(env._left_x_position) == (1 - 250) % 256


def test_y_position_above_viewport_uses_overflow_rule():
    env = _make_env_without_init()
    env.ram[0x00B5] = 0  # above viewport
    env.ram[0x03B8] = 10
    # Current implementation uses uint8 math via self._y_pixel, which wraps.
    assert int(env._y_position) == (255 + (255 - 10)) % 256


def test_y_position_normal_viewport_inverts_pixel():
    env = _make_env_without_init()
    env.ram[0x00B5] = 1
    env.ram[0x03B8] = 10
    assert env._y_position == 245


def test_player_status_mapping_default_and_known_values():
    env = _make_env_without_init()
    env.ram[0x0756] = 0
    assert env._player_status == "small"
    env.ram[0x0756] = 1
    assert env._player_status == "tall"
    env.ram[0x0756] = 9
    assert env._player_status == "fireball"


def test_is_dead_and_dying_logic():
    env = _make_env_without_init()
    env.ram[0x000E] = 0x0b
    assert bool(env._is_dying) is True
    env.ram[0x000E] = 0x06
    env.ram[0x00B5] = 1
    assert bool(env._is_dead) is True


def test_is_game_over_when_life_is_ff():
    env = _make_env_without_init()
    env.ram[0x075A] = 0xFF
    assert bool(env._is_game_over) is True


def test_is_busy_recognizes_busy_states():
    env = _make_env_without_init()
    env.ram[0x000E] = 0x00
    assert bool(env._is_busy) is True
    env.ram[0x000E] = 0x08
    assert bool(env._is_busy) is False


def test_is_world_over_reads_gameplay_mode():
    env = _make_env_without_init()
    env.ram[0x0770] = 2
    assert bool(env._is_world_over) is True
    env.ram[0x0770] = 1
    assert bool(env._is_world_over) is False


def test_is_stage_over_requires_stage_over_enemy_and_float_state():
    env = _make_env_without_init()

    # set a stage over enemy but wrong float state => False
    env.ram[0x0016] = 0x31
    env.ram[0x001D] = 0
    assert bool(env._is_stage_over) is False

    # set correct float state => True
    env.ram[0x001D] = 3
    assert bool(env._is_stage_over) is True


def test_flag_get_true_if_world_or_stage_over():
    env = _make_env_without_init()
    env.ram[0x0770] = 2
    assert bool(env._flag_get) is True
    env.ram[0x0770] = 1
    env.ram[0x0016] = 0x31
    env.ram[0x001D] = 3
    assert bool(env._flag_get) is True


def test_is_single_stage_env_property():
    env = _make_env_without_init()
    assert env.is_single_stage_env is False
    env._target_world = 1
    env._target_area = 2
    assert env.is_single_stage_env is True
