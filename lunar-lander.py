import gymnasium as gym

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import random

VERBOSE = False


def plot_rewards(agent_return, num_episodes, window):
    num_intervals = int(num_episodes / window)
    mean = []
    for interval in range(num_intervals): mean.append(round(np.mean(agent_return[interval * 100 : (interval + 1) * 100]), 1))
    plt.plot(range(0, num_episodes, window), mean)

    plt.xlabel("Episodes")
    plt.ylabel("Reward per {} episodes".format(window))
    plt.legend("QLearner", loc="lower right")
    plt.show()


def reshape_obs(observation):
    if VERBOSE: print("OBS:", observation)
    discrete_state = [int(observation[0]), int(observation[1]), int(observation[4]), int(observation[5]), int(observation[6])]
    return f'{np.asarray(discrete_state)}'

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")#, render_mode="human")

    # Hyper params
    epsilon = 0.2
    min_epsilon = 0.01
    epsilon_decay = 0.999
    gamma = 0.99
    alpha = 0.1

    num_episodes = 10000

    observation, info = env.reset()
    q_table = q = defaultdict(lambda: np.zeros(env.action_space.n))
    num_actions = 0

    episode_rewards = [0.0]
    for episode in range(num_episodes):
        
        while True:
            # Find new action with epsilon-greedy
            action = np.argmax(q_table[reshape_obs(observation)])
            if random.random() < epsilon:
                if epsilon > min_epsilon: epsilon *= epsilon_decay
                action = random.randint(0, 3)
            
            # Take action and get reward
            new_observation, reward, terminated, truncated, info = env.step(action)
            episode_rewards[-1] += reward
            num_actions += 1
            
            # Update Q-Table
            if terminated: q_table[reshape_obs(observation)][action] += alpha * (reward + gamma * np.max(q_table[reshape_obs(new_observation)]) - q_table[reshape_obs(observation)][action])
            else: q_table[reshape_obs(observation)][action] += alpha * (reward - q_table[reshape_obs(observation)][action])
            
            if terminated or truncated:
                episode_rewards.append(0.0)
                observation, info = env.reset()
                break

    plot_rewards(episode_rewards, num_episodes, 100)

    env.close()