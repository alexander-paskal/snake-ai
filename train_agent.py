"""
Train SB3 Wrapper that actually works. Pip install Gym-Snake
Might have to fix a couple of errors

Add the Tensorboard 
"""
from typing import Callable
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import BaseCallback
from snake_env import SnakeEnvWrapper
from argparse import ArgumentParser
from pathlib import Path
import os



def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def log10_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Logarithmic Schedule that decays the learning rate logarithmically
    0%  = IV
    10% = IV * 0.1
    20% = IV * 0.001
    ...

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return initial_value * (10**-(10-10*progress_remaining))

    return func

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.random.random()
        self.logger.record("random_value", value)
        return True






if __name__ == "__main__":
    # Initialize the custom environment
    n_envs = 4
    agent = "Dummy"
    agentsdir = Path("agents")
    agentsdir.mkdir(exist_ok=True)

    # env = SubprocVecEnv([lambda: gym.make("snake-sb3") for _ in range(n_envs)])  # uncomment for multithread training
    env = SnakeEnvWrapper(grid_size=[8,8])

    # Configure the logger for TensorBoard
    log_dir = "./tensorboard_logs/"
    new_logger = configure(log_dir, ["tensorboard"])

    # Create the PPO model
    if not os.path.exists(f"{agent}.zip"):
        print("Birthing", agent)
        model = PPO("CnnPolicy", env, verbose=2, learning_rate=log10_schedule(0.001), tensorboard_log=log_dir)
    else:
        print("Loading", agent)
        model = PPO.load(f"agentsdir / {agent}.zip", env=env, learning_rate=log10_schedule(0.001), tensorboard_log=log_dir)

    model.set_logger(new_logger)

    # Train the model
    model.learn(total_timesteps=6_000_000, progress_bar=True)
    model.save(agentsdir / f"{agent}")
    model.policy.save(f"{agent}_policy")
    torch.save(model.policy.state_dict(), agentsdir / f"{agent}_policy.pt")
    
    env.close()