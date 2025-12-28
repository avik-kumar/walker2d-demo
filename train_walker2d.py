import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor
from CustomReward import CustomReward

# Environment setup and normalization
def make_env():
    env = gym.make("Walker2d-v4", healthy_angle_range=(-0.4,0.4), max_episode_steps=1500)
    # Here, we wrap the env with a custom reward signal, encouraging a more upright torso.
    # Removing this generates the default reward signal and thus the default behavior (overly leaning forward).
    env = CustomReward(env)
    return env

env = DummyVecEnv([make_env])
env = VecMonitor(env)

env = VecNormalize(
    env,
    norm_obs=True,
    norm_reward=True,
    clip_obs=10.0
)

checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path="./checkpoints/",
    name_prefix="ppo_walker2d"
)

# PPO model
model = PPO(
    policy="MlpPolicy",
    env=env,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    learning_rate=3e-4,
    ent_coef=0.0,
    verbose=1,
    tensorboard_log="./tensorboard_logs/",
    device="cpu"
)

# Training + Save Data
TOTAL_TIMESTEPS = 800_000

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_callback
)

model.save("ppo_walker2d_final")
env.save("vecnormalize.pkl")

env.close()