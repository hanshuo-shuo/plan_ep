import gymnasium as gym
from stable_baselines3 import DQN
import cellworld_gym as cwg

def evaluate():
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
    runs = 100
    # show progress bar
    import tqdm
    # eval for 200 runs
    for step in tqdm.tqdm(range(runs)):
        rewards = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, tr, info = env.step(action)
            rewards += reward
            done = done or tr
            if done:
                if rewards > 0:
                    success += 1
                    print(rewards)
                done = False
                obs, _ = env.reset()
                break
    print(f"Success rate: {success/runs}")

if __name__ == "__main__":
    evaluate()