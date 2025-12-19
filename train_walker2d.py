import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

# This is the custom reward class that is a subclass of gymnasium wrappers. We use this to customize 
# the 'step' method, which shapes the reward signal towards new desired behaviors. The behaviors I 
# wanted to encourage were a more upright torso, legs centered around the torso (via hip penalty),
# as well as less torque in joints for stable movement.
class CustomReward(gym.Wrapper):
    def __init__(self, env, torso_weight=1.0, knee_weight=0.3, symmetry_weight=0.1):
        super().__init__(env)
        self.torso_weight = torso_weight
        self.knee_weight = knee_weight
        self.symmetry_weight = symmetry_weight

    def step(self,action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        torso_angle = obs[1]
        left_knee  = obs[3]
        right_knee = obs[6]
        left_hip   = obs[2]
        right_hip  = obs[5]

        torso_penalty = self.torso_weight * torso_angle**2
        knee_penalty = self.knee_weight * (left_knee**2 + right_knee**2)
        symmetry_penalty = self.symmetry_weight * (
            (left_knee - right_knee)**2 +
            (left_hip  - right_hip)**2
        )

        # Shaped reward
        shaped_reward = reward - torso_penalty - knee_penalty - symmetry_penalty

        return obs, shaped_reward, terminated, truncated, info

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