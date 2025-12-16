import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import RecordVideo, TimeLimit

def record_video():
    # Environment setup (with video)
    def make_env():
        env = gym.make("Walker2d-v4", render_mode="rgb_array")
        env = TimeLimit(env, max_episode_steps=3000)

        env = RecordVideo(
            env,
            video_folder="./videos/",
            episode_trigger=lambda episode_id: True,  # record first episode
            name_prefix="ppo_walker2d"
        )
        return env

    env = DummyVecEnv([make_env])


    # Get stats from pickle
    env = VecNormalize.load("vecnormalize.pkl", env)
    env.training = False
    env.norm_reward = False


    # Load model from zip folder
    model = PPO.load("ppo_walker2d_final", env=env, device="cpu")


    # Run episode
    obs = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

    env.close()

if __name__ == "__main__":
    record_video()