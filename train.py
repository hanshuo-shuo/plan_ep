import gymnasium as gym
import numpy as np
from collections import deque
import torch
import argparse
from buffer import ReplayBuffer
import glob
from utils import save, collect_random, collect_dqn_expert
from agent import IQL
import cellworld_gym as cwg
import random
from stable_baselines3 import DQN

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="IQL-discrete", help="Run name, default: SAC")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes, default: 100")
    parser.add_argument("--buffer_size", type=int, default=100_000, help="Maximal training dataset size, default: 100_000")
    parser.add_argument("--seed", type=int, default=123, help="Seed, default: 1")
    parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size, default: 256")

    args = parser.parse_args()
    return args

def train(config):
    torch.manual_seed(config.seed)
    env = gym.make("CellworldBotEvade-v0",
                   world_name="21_05",
                   use_lppos=False,
                   use_predator=True,
                   max_step=300,
                   time_step=0.25,
                   render=False,
                   real_time=False,
                   reward_function=cwg.Reward({"puffed": -1, "finished": 1}))
    expert_planner = DQN.load("DQNmouse_control.zip", env=env)
    initial_planning_rate = 1.0
    final_planning_rate = 0.0
    planning_rate_decay = (initial_planning_rate - final_planning_rate) / config.episodes
    planning_rate = initial_planning_rate

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    steps = 0
    average10 = deque(maxlen=10)

    agent = IQL(state_size=env.observation_space.shape[0],
                action_size=env.action_space.n,
                device=device, learning_rate=1e-4)

    buffer = ReplayBuffer(buffer_size=config.buffer_size, batch_size=config.batch_size, device=device)

    # collect_dqn_expert(env, buffer, model=expert_planner, num_samples=1000)
    collect_random(env, buffer, num_samples=1000)
    print("Buffer size: ", buffer.__len__())
    reward_list = []
    planning_rate_list = []
    for i in range(1, config.episodes + 1):
        state, _ = env.reset()
        episode_steps = 0
        rewards = 0
        while True:
            if random.uniform(0, 1) < planning_rate:
                action, _= expert_planner.predict(state)
            else:
                action = agent.get_action(state)
            steps += 1
            next_state, reward, done, tr, _ = env.step(action)
            done = done or tr
            buffer.add(state, action, reward, next_state, done)
            # sample from buffer with bias towards negative rewards
            sampled_data = buffer.sample(type='bias')
            # learn from the sampled data
            policy_loss, critic1_loss, critic2_loss, value_loss = agent.learn(sampled_data)
            state = next_state
            rewards += reward
            episode_steps += 1
            if done:
                break
        reward_list.append(rewards)
        planning_rate_list.append(planning_rate)
        # update planning rate
        planning_rate = max(final_planning_rate, planning_rate - planning_rate_decay)
        print("Planning rate: ", planning_rate)

        average10.append(rewards)
        print("Episode: {} | Reward: {} | Polciy Loss: {} | Steps: {}".format(i, rewards, policy_loss, steps, ))

        if i % config.save_every == 0:
            save(config, save_name="IQL", model=agent.actor_local, ep=3)
        # save the reward list
    np.save("reward_list_2.npy", reward_list)


def eval_agent():
    env = gym.make("CellworldBotEvade-v0",
                   world_name="21_05",
                   use_lppos=False,
                   use_predator=True,
                   max_step=300,
                   time_step=0.25,
                   render=True,
                   real_time=True,
                   reward_function=cwg.Reward({"puffed": -1, "finished": 1}))
    obs, _ = env.reset()
    agent = IQL(state_size=env.observation_space.shape[0],
                action_size=env.action_space.n,
                device="cpu")
    agent.actor_local.load_state_dict(torch.load("trained_models/IQL-discreteIQL2.pth"))
    for step in range(1000):
        action = agent.get_action(obs, eval=True)
        obs, reward, done, tr, info = env.step(action)
        done = done or tr
        if done:
            obs, _ = env.reset()
def plot():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    # Load the reward list
    reward_list = np.load("reward_list_2.npy")
    # Convert to pandas Series for convenience
    reward_series = pd.Series(reward_list)
    # Calculate the moving average with a window size of 10
    smooth_reward_list = reward_series.rolling(window=50).mean()
    # Create a new figure
    plt.figure(figsize=(10, 6))
    # Plot the original reward list in light gray
    plt.plot(reward_list, color='lightgray', label='Original')
    # Plot the smoothed reward list in blue
    plt.plot(smooth_reward_list, color='blue', label='Smoothed')
    # Add labels and title
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward over time')
    # label the last mean reward as "Mean Reward: " and the actual value
    plt.text(len(reward_list), smooth_reward_list.iloc[-1], f'Mean Reward: {smooth_reward_list.iloc[-1]:.2f}')
    # Add a legend
    plt.legend()
    # Add a grid
    plt.grid(True)
    # Show the plot
    plt.show()

def get_success_rate(runs=100):
    env = gym.make("CellworldBotEvade-v0",
                   world_name="21_05",
                   use_lppos=False,
                   use_predator=True,
                   max_step=300,
                   time_step=0.25,
                   render=False,
                   real_time=False,
                   reward_function=cwg.Reward({"puffed": -1, "finished": 1}))
    obs, _ = env.reset()
    model = DQN.load("DQNmouse_control.zip", env=env)
    done = False
    success = 0
    for step in range(runs):
        while not done:
            rewards = 0
            action, _ = model.predict(obs)
            obs, reward, done, tr, info = env.step(action)
            rewards += reward
            done = done or tr
            if done:
                if rewards > 0:
                    success += 1
                done = False
                obs, _ = env.reset()
    return success / runs


if __name__ == "__main__":
    # eval_agent()
    # config = get_config()
    # train(config)
    plot()