"""Registration code of Gym environments in this package.

Compatibility:
    Gym 0.26.x still references the `np.bool8` alias, which was removed in
    NumPy 2.0. To keep this package working on Python 3.12 + NumPy 2.x, we
    provide a small alias during import.
"""

from __future__ import annotations

import numpy as np

# NumPy 2.0 removed `np.bool8`; Gym<=0.26 still references it.
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

from .smb_env import SuperMarioBrosEnv
from .smb_random_stages_env import SuperMarioBrosRandomStagesEnv
from ._registration import make


# define the outward facing API of this package
__all__ = [
    make.__name__,
    SuperMarioBrosEnv.__name__,
    SuperMarioBrosRandomStagesEnv.__name__,
]
