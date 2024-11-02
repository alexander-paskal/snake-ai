"""
Train SB3 Wrapper that actually works. Pip install Gym-Snake
Might have to fix a couple of errors

Add the Tensorboard 
"""
from typing import Callable, Optional, Type, List, Dict, Union
import gymnasium as gym
import numpy as np
import random
from gymnasium.spaces import Discrete, Box
import torch
from stable_baselines3 import A2C, SAC, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo.policies import CnnPolicy
from snake_env import SnakeEnvWrapper
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import shutil




if __name__ == "__main__":
    # Initialize the custom environment
    n_envs = 4  # Set the number of parallel environments
    agent = "Johnny-Dense-1"
    agentdir = Path("agents")
    outputdir = Path("images")

    # env = SubprocVecEnv([lambda: gym.make("snake-sb3") for _ in range(n_envs)])
    env = SnakeEnvWrapper(grid_size=[8,8], max_steps=500)


    model = PPO.load(agentdir / f"{agent}.zip", env=env)


    obs, _ = env.reset()
    done = False
    env.render()
    i = 0
    grids = [env.controller.grid.grid]

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, done, _, info = env.step(action)
        grids.append(env.controller.grid.grid.copy())
        env.render()
        i += 1

    # create animation
    print("Creating animation")
    fig, ax = plt.subplots()
    implot = ax.imshow(grids[0], animated=True)

    def update(frame):
        im = grids[frame]
        implot.set_array(im)
        return [implot]

    ani = animation.FuncAnimation(fig,update, frames=len(grids), blit=True)
    print("saving animation")
    ani.save(filename=f"{agent}.gif", writer="pillow", fps=5)
    plt.show()