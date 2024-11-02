"""
Train SB3 Wrapper that actually works. Pip install Gym-Snake
Might have to fix a couple of errors

Add the Tensorboard 
"""
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from stable_baselines3.common.env_checker import check_env
from gym_snake import SnakeEnv



_ID = 0
class SnakeEnvWrapper(SnakeEnv, gym.Env):


    def __init__(self, *args, max_steps=200, **kwargs):
        # new = copy.deepcopy(kwargs)
        # new["grid_size"] = [15, 15]
        # new["n_snakes"] = 1

        super().__init__(*args, **kwargs)

        global _ID
        self._ID = _ID
        _ID += 1
        
        # print(f"{self._ID} - constructed!")
        # denote max steps
        self.max_steps = 200
        self.n_steps = 0
        self.n_rewards = 0
        # infer obs space
        sample_obs = self.process_obs(super().reset())
        self.observation_space =  Box(
            0,
            255,
            sample_obs.shape,
            np.uint8
        )
    
    def step(self, action):
        # try:
        last_obs, rewards, done, info = super().step(action)
        # print(f"{self._ID} - step {self.n_steps}")

        self.n_steps += 1
        if self.n_steps >= self.max_steps:
            # print(f"{self._ID} - max steps reached!")
            done = True
        
        if rewards == 1:
            self.n_rewards += rewards  # increment reward count

        dense_rew = self.dense_distance_reward()
        # print("Dense Rew:", dense_rew)
        rewards += dense_rew
        return self.process_obs(last_obs), rewards, done, False, info 
        # except Exception as e:
        #     return self.last_obs, rewards, done, True, info 

    def reset(self, seed=None):
        obs = super().reset()
        # print(f"{self._ID} - resetting")
        self.n_steps = 0
        return self.process_obs(obs), {}

    def process_obs(self, obs):
        if obs.max() <= 1:
            obs = obs * 255
        obs = obs.astype(np.uint8)
        return np.transpose(obs, (2, 0, 1))
    
    def dense_distance_reward(self):
        grid = self.controller.grid
        snake = self.controller.snakes[0]
        food = grid.food_coord

        if snake is None:
            return 0
        
        dist = abs(food[0] - snake.head[0]) + abs(food[1] - snake.head[1])
        rew = 1/dist * 0.01
        return rew
    
    


from gymnasium.envs.registration import register
register("snake-sb3", entry_point="__main__:SnakeEnvWrapper")


if __name__ == "__main__":
    env = SnakeEnvWrapper()
    check_env(env)




