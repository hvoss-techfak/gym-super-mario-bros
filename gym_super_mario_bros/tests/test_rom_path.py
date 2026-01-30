import os
import pytest

from gym_super_mario_bros._roms.rom_path import rom_path


def test_rom_path_typecheck_lost_levels():
    with pytest.raises(TypeError):
        rom_path(lost_levels="nope", rom_mode="vanilla")


@pytest.mark.parametrize(
    "lost_levels,rom_mode,expected_basename",
    [
        (False, "vanilla", "super-mario-bros.nes"),
        (False, "downsample", "super-mario-bros-downsample.nes"),
        (False, "pixel", "super-mario-bros-pixel.nes"),
        (False, "rectangle", "super-mario-bros-rectangle.nes"),
        (True, "vanilla", "super-mario-bros-2.nes"),
        (True, "downsample", "super-mario-bros-2-downsample.nes"),
    ],
)
def test_rom_path_supported_modes_return_existing_files(lost_levels, rom_mode, expected_basename):
    p = rom_path(lost_levels=lost_levels, rom_mode=rom_mode)
    assert os.path.isabs(p)
    assert os.path.basename(p) == expected_basename
    assert os.path.exists(p)


def test_rom_path_invalid_rom_mode_raises_value_error():
    with pytest.raises(ValueError):
        rom_path(lost_levels=False, rom_mode="does-not-exist")
