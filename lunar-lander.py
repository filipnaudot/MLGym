import gymnasium as gym

import numpy as np
import collections
import matplotlib.pyplot as pp
import random

    

if __name__ == "__main__":
    env = gym.make("LunarLander-v2", render_mode="human")
    observation, info = env.reset()

    rewards = []
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(random.randint(0,3))
        rewards.append(reward)

        if terminated or truncated:
            observation, info = env.reset()


    # Plot rewards
    pp.plot(rewards, 'x')
    pp.show()


    env.close()