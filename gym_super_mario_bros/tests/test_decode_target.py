import pytest

from gym_super_mario_bros._roms.decode_target import decode_target


def test_decode_target_requires_bool_lost_levels():
    with pytest.raises(TypeError):
        decode_target((1, 1), lost_levels="False")


def test_decode_target_none_target_returns_triplet_of_none():
    assert decode_target(None, lost_levels=False) == (None, None, None)


def test_decode_target_requires_tuple_target():
    with pytest.raises(TypeError):
        decode_target([1, 1], lost_levels=False)


@pytest.mark.parametrize("world", [0, 9, 999])
def test_decode_target_world_bounds_smb1(world):
    with pytest.raises(ValueError):
        decode_target((world, 1), lost_levels=False)


@pytest.mark.parametrize("world", [0, 13, 999])
def test_decode_target_world_bounds_lost_levels(world):
    with pytest.raises(ValueError):
        decode_target((world, 1), lost_levels=True)


@pytest.mark.parametrize("stage", [0, 5, 42])
def test_decode_target_stage_bounds(stage):
    with pytest.raises(ValueError):
        decode_target((1, stage), lost_levels=False)


def test_decode_target_adjusts_area_for_smb1_world_1_stage_2_plus():
    assert decode_target((1, 1), lost_levels=False) == (1, 1, 1)
    assert decode_target((1, 2), lost_levels=False) == (1, 2, 3)
    assert decode_target((1, 4), lost_levels=False) == (1, 4, 5)


def test_decode_target_adjusts_area_for_smb1_world_3_does_not_shift():
    assert decode_target((3, 2), lost_levels=False) == (3, 2, 2)


def test_decode_target_lost_levels_world_1_shifts_for_stage_2_plus():
    assert decode_target((1, 1), lost_levels=True) == (1, 1, 1)
    assert decode_target((1, 2), lost_levels=True) == (1, 2, 3)


def test_decode_target_lost_levels_worlds_5plus_not_supported():
    with pytest.raises(ValueError):
        decode_target((5, 1), lost_levels=True)
