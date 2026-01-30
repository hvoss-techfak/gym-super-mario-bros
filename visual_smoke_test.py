"""Visual smoke test for gym-super-mario-bros.

This script runs the Super Mario Bros environment with random actions.
It:
- prints reward/termination info to the console
- displays the latest frame in a window (via Matplotlib)

Notes:
- This uses `env.render()` if available; otherwise it falls back to the
  observation returned by `reset/step`.
- Quit by closing the window or pressing `q` in the Matplotlib window.
"""

from __future__ import annotations

import time

import numpy as np

from gym_super_mario_bros import SuperMarioBrosEnv
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT


def _as_rgb_frame(frame) -> np.ndarray:
    """Normalize frames into uint8 RGB HxWx3 for display."""
    if frame is None:
        raise ValueError("No frame available to render")

    frame = np.asarray(frame)

    # Common outputs:
    # - (240, 256, 3) uint8
    # - (240, 256) grayscale
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)

    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"Unexpected frame shape: {frame.shape}")

    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    return frame


def _resolve_action_set(name: str) -> list[list[str]]:
    name = name.lower().strip()
    if name in {"right", "right_only"}:
        return RIGHT_ONLY
    if name in {"simple", "simple_movement"}:
        return SIMPLE_MOVEMENT
    if name in {"complex", "complex_movement"}:
        return COMPLEX_MOVEMENT
    raise ValueError(f"Unknown action set: {name!r} (use: right|simple|complex)")


def main(
    steps: int = 2000,
    fps: float = 60.0,
    seed: int = 0,
    action_set: str = "simple",
) -> int:
    # Lazy imports so the library itself doesn't require these extras.
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "This script requires Matplotlib. Install it with:\n"
            "  pip install matplotlib\n\n"
            f"Original import error: {e}"
        )

    try:
        from nes_py.wrappers import JoypadSpace
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "This script requires nes-py wrappers (JoypadSpace).\n"
            f"Original import error: {e}"
        )

    actions = [['right'],]

    env = JoypadSpace(SuperMarioBrosEnv(), actions)

    # Matplotlib window setup.
    plt.ion()
    fig, ax = plt.subplots(num="gym-super-mario-bros: random policy", clear=True)
    ax.set_axis_off()

    img_artist = None

    try:
        obs, info = env.reset(seed=seed)
        action_space_n = getattr(getattr(env, "action_space", None), "n", None)
        print(
            f"reset: obs={getattr(obs, 'shape', None)} action_space_n={action_space_n} "
            f"info_keys={list(info.keys())}"
        )

        #delay_s = 0.0 if fps <= 0 else (1.0 / fps)
        total_reward = 0

        for t in range(steps):
            a = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(a)
            total_reward += reward
            # Try env.render() first (may return None depending on nes-py setup).
            frame = None
            try:
                frame = env.render()
            except Exception:
                frame = None
            if frame is None:
                frame = obs

            rgb = _as_rgb_frame(frame)

            if img_artist is None:
                img_artist = ax.imshow(rgb, interpolation="nearest")
            else:
                img_artist.set_data(rgb)

            # Some backends don't deliver keypresses without an active event loop.
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)

            meaning = None
            try:
                meaning = env.get_action_meanings()[a]
            except Exception:
                meaning = str(actions[a]) if 0 <= a < len(actions) else str(a)

            # print(
            #     f"t={t:05d} action={a:2d} meaning={meaning!s:>12} reward={reward:7.3f} "
            #     f"terminated={terminated} truncated={truncated} "
            #     f"x_pos={info.get('x_pos')} world={info.get('world')} stage={info.get('stage')}"
            # )

            if not plt.fignum_exists(fig.number):
                break

            if terminated or truncated:
                obs, info = env.reset()
                print(f"episode ended -> reset. Total reward: {total_reward:.3f}")

            #if delay_s:
            #    time.sleep(delay_s)

    finally:
        try:
            env.close()
        finally:
            try:
                plt.close("all")
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
