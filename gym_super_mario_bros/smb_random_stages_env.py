"""An OpenAI Gym Super Mario Bros. environment that randomly selects levels."""
import gymnasium as gym
import numpy as np
from .smb_env import SuperMarioBrosEnv


class SuperMarioBrosRandomStagesEnv(gym.Env):
    """A Super Mario Bros. environment that randomly selects levels."""

    # relevant meta-data about the environment
    metadata = SuperMarioBrosEnv.metadata

    # the legal range of rewards for each step
    reward_range = SuperMarioBrosEnv.reward_range

    # observation/action spaces must be Gymnasium spaces
    observation_space = gym.spaces.Box(low=0, high=255, shape=(240, 256, 3), dtype=np.uint8)
    action_space = gym.spaces.Discrete(256)

    def __init__(self, rom_mode='vanilla', stages=None):
        """
        Initialize a new Super Mario Bros environment.

        Args:
            rom_mode (str): the ROM mode to use when loading ROMs from disk
            stages (list): select stages at random from a specific subset

        Returns:
            None

        """
        # Dedicated RNG for stage selection.
        # Use RandomState to preserve historical determinism expected by tests
        # (e.g., seed=1 -> world=6, stage=4).
        self._stage_rng = np.random.RandomState()
        # Expose for legacy / unit tests.
        self.np_random = self._stage_rng
        # setup the environments
        self.envs = []
        # iterate over the worlds in the game, i.e., {1, ..., 8}
        for world in range(1, 9):
            # append a new list to put this world's stages into
            self.envs.append([])
            # iterate over the stages in the world, i.e., {1, ..., 4}
            for stage in range(1, 5):
                # create the target as a tuple of the world and stage
                target = (world, stage)
                # create the environment with the given ROM mode
                env = SuperMarioBrosEnv(rom_mode=rom_mode, target=target)
                # add the environment to the stage list for this world
                self.envs[-1].append(env)
        # create a placeholder for the current environment
        self.env = self.envs[0][0]
        # create a placeholder for the image viewer to render the screen
        self.viewer = None
        # create a placeholder for the subset of stages to choose
        self.stages = stages

    @property
    def screen(self):
        """Return the screen from the underlying environment"""
        return self.env.screen

    def seed(self, seed=None):
        """Legacy seeding API."""
        if seed is None:
            return []
        self._stage_rng.seed(seed)
        return [seed]

    def reset(self, seed=None, options=None, return_info=None):
        """Reset the env and return (obs, info) per Gymnasium's API."""
        # Gymnasium bookkeeping (may overwrite `self.np_random`).
        super().reset(seed=seed, options=options)
        # Restore legacy attribute and seed our dedicated RNG.
        self.np_random = self._stage_rng
        if seed is not None:
            self.seed(seed)

        # Get the collection of stages to sample from
        stages = self.stages
        if options is not None and 'stages' in options:
            stages = options['stages']

        # Select a random level
        if stages is not None and len(stages) > 0:
            level = self._stage_rng.choice(stages)
            if not isinstance(level, str):
                level = str(level)
            world, stage = level.split('-')
            world = int(world) - 1
            stage = int(stage) - 1
        else:
            world = int(self._stage_rng.randint(1, 9)) - 1
            stage = int(self._stage_rng.randint(1, 5)) - 1

        # Set the environment based on the world and stage.
        self.env = self.envs[world][stage]
        # reset the environment
        return self.env.reset(
            seed=seed,
            options=options,
            return_info=return_info
        )

    def step(self, action):
        """
        Run one frame of the NES and return the relevant observation data.

        Args:
            action (byte): the bitmap determining which buttons to press

        Returns:
            a tuple of:
            - state (np.ndarray): next frame as a result of the given action
            - reward (float) : amount of reward returned after given action
            - done (boolean): whether the episode has ended
            - info (dict): contains auxiliary diagnostic information

        """
        return self.env.step(action)

    def close(self):
        """Close the environment."""
        # make sure the environment hasn't already been closed
        if self.env is None:
            raise ValueError('env has already been closed.')
        # iterate over each list of stages
        for stage_lists in self.envs:
            # iterate over each stage
            for stage in stage_lists:
                # close the environment
                stage.close()
        # close the environment permanently
        self.env = None
        # if there is an image viewer open, delete it
        if self.viewer is not None:
            self.viewer.close()

    def render(self, mode='human'):
        """
        Render the environment.

        Args:
            mode (str): the mode to render with:
            - human: render to the current display
            - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
              representing RGB values for an x-by-y pixel image

        Returns:
            a numpy array if mode is 'rgb_array', None otherwise

        """
        return SuperMarioBrosEnv.render(self, mode=mode)

    def get_keys_to_action(self):
        """Return the dictionary of keyboard keys to actions."""
        return self.env.get_keys_to_action()

    def get_action_meanings(self):
        """Return the list of strings describing the action space actions."""
        return self.env.get_action_meanings()


# explicitly define the outward facing API of this module
__all__ = [SuperMarioBrosRandomStagesEnv.__name__]
