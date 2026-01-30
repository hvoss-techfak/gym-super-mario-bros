from pathlib import Path

import pytest

from gym_super_mario_bros._roms.rom_compat import ensure_rom_ok, read_ines_header


def _write_rom(path: Path, *, prg_units: int = 1, chr_units: int = 1, flags6: int = 0, zero_fill_ok: bool = True):
    header = bytearray(16)
    header[0:4] = b"NES\x1a"
    header[4] = prg_units
    header[5] = chr_units
    header[6] = flags6
    # bytes 7-10 left as zero
    if zero_fill_ok:
        header[11:16] = b"\x00\x00\x00\x00\x00"
    else:
        header[11:16] = b"\x01\x00\x00\x00\x00"

    # add some payload bytes as well
    path.write_bytes(bytes(header) + b"\x00" * 64)


def test_read_ines_header_missing_file_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        read_ines_header(tmp_path / "missing.nes")


def test_read_ines_header_too_small_raises(tmp_path: Path):
    p = tmp_path / "tiny.nes"
    p.write_bytes(b"123")
    with pytest.raises(ValueError):
        read_ines_header(p)


def test_read_ines_header_bad_magic_raises(tmp_path: Path):
    p = tmp_path / "bad.nes"
    p.write_bytes(b"NOPE" + b"\x00" * 32)
    with pytest.raises(ValueError):
        read_ines_header(p)


def test_read_ines_header_non_zero_fill_raises(tmp_path: Path):
    p = tmp_path / "badfill.nes"
    _write_rom(p, zero_fill_ok=False)
    with pytest.raises(ValueError):
        read_ines_header(p)


def test_read_ines_header_parses_sizes_as_ints(tmp_path: Path):
    p = tmp_path / "ok.nes"
    _write_rom(p, prg_units=2, chr_units=3, flags6=0b0000_0100)
    h = read_ines_header(p)
    assert h.prg_rom_size_16kb_units == 2
    assert h.chr_rom_size_8kb_units == 3
    assert h.has_trainer is True
    assert h.prg_rom_size_bytes == 2 * 16 * 1024
    assert h.chr_rom_size_bytes == 3 * 8 * 1024


def test_ensure_rom_ok_returns_string_path(tmp_path: Path):
    p = tmp_path / "ok.nes"
    _write_rom(p)
    assert ensure_rom_ok(p) == str(p)
