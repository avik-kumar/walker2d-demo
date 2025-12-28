import gymnasium as gym

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