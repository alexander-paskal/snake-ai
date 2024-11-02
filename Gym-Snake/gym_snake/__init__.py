from gymnasium.envs.registration import register
from gym_snake.envs import SnakeEnv, SnakeExtraHardEnv

register(
    id='snake-v0',
    entry_point='gym_snake.envs:SnakeEnv',
)
register(
    id='snake-plural-v0',
    entry_point='gym_snake.envs:SnakeExtraHardEnv',
)
