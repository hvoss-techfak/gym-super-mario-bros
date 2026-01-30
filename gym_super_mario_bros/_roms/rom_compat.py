"""Compatibility helpers for loading ROM files.

Why this exists:
    Recent NumPy versions are stricter about mixing numpy scalar types
    (e.g., np.uint8) with Python ints, and some upstream `nes-py` versions
    do arithmetic in a way that can overflow when a numpy scalar sneaks into
    address calculations.

This project only needs a path to a valid `.nes` file. This module provides
small helpers that validate the header and exposes sizes as plain Python ints.

We intentionally *don't* re-implement a full NES ROM parser; we only cover:
- magic bytes
- PRG/CHR size bytes
- optional trainer flag

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_INES_MAGIC = b"NES\x1a"


@dataclass(frozen=True)
class INESHeader:
    """Parsed subset of an iNES header."""

    prg_rom_size_16kb_units: int
    chr_rom_size_8kb_units: int
    flags6: int

    @property
    def has_trainer(self) -> bool:
        return bool(self.flags6 & 0b0000_0100)

    @property
    def prg_rom_size_bytes(self) -> int:
        # Byte 4 is the number of 16KB PRG banks.
        return self.prg_rom_size_16kb_units * 16 * 1024

    @property
    def chr_rom_size_bytes(self) -> int:
        # Byte 5 is the number of 8KB CHR banks.
        return self.chr_rom_size_8kb_units * 8 * 1024


def read_ines_header(rom_path: str | Path) -> INESHeader:
    """Read and validate the first 16 bytes of an iNES ROM.

    Returns:
        INESHeader with fields as Python ints (never numpy scalars).

    Raises:
        ValueError: if the file is missing / too small / invalid header.
    """

    p = Path(rom_path)
    if not p.exists():
        raise ValueError(f"rom_path points to non-existent file: {p}.")

    data = p.read_bytes()
    if len(data) < 16:
        raise ValueError("ROM file is too small to contain an iNES header.")
    if data[:4] != _INES_MAGIC:
        raise ValueError("ROM missing magic number in header.")

    # cast to Python int explicitly
    prg = int(data[4])
    chr_ = int(data[5])
    flags6 = int(data[6])

    # bytes 11-15 should be zero-filled
    if any(b != 0 for b in data[11:16]):
        raise ValueError("ROM header zero fill bytes are not zero.")

    return INESHeader(prg_rom_size_16kb_units=prg, chr_rom_size_8kb_units=chr_, flags6=flags6)


def ensure_rom_ok(rom_path: str | Path) -> str:
    """Validate ROM header and return the path as a string.

    This is used as a fast-fail guard *before* we hand the path to `nes-py`.
    """

    _ = read_ines_header(rom_path)
    return str(rom_path)
