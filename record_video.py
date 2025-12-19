import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.vec_env import VecVideoRecorder

env = DummyVecEnv([lambda: gym.make("Walker2d-v4", render_mode="rgb_array", max_episode_steps=1500)])
env = VecNormalize.load("vecnormalize.pkl", env)
env.training = False
env.norm_reward = False

env = VecVideoRecorder(
    env,
    "./videos/",
    record_video_trigger=lambda x: x == 0,  # record first episode
    video_length=1500,  # maximum length
    name_prefix="ppo_walker2d"
)

model = PPO.load("ppo_walker2d_final", env=env, device="cpu")

obs = env.reset()
done = [False]
while not done[0]:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

env.close()