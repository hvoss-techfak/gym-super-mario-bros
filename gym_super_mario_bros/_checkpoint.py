"""Checkpoint support for gym-super-mario-bros.

nes-py exposes a single internal backup slot via `NESEnv._backup()`/
`NESEnv._restore()`. This module builds an *arbitrary* multi-checkpoint API by
copying emulator snapshots into temporary emulator instances.

The checkpoint payload is intentionally opaque. It may not be portable across
nes-py versions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SmbCheckpoint:
    """Opaque checkpoint that can be saved/loaded by `SuperMarioBrosEnv`."""

    rom_path: str

    # Target to guard against restoring into the wrong env.
    target_world: Optional[int]
    target_stage: Optional[int]
    target_area: Optional[int]

    # Env bookkeeping.
    time_last: int
    x_position_last: int

    # Snapshot bytes (implementation detail).
    _payload: bytes
