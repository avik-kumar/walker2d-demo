# Walker2d Experimentation w/Custom Reward Policies

## Objective
The main objective of this mini-project was to train a humanoid policy using RL techniques and working a physics engine library. Since a default reward policy is usually not ideal, part of this project was to develop a custom policy to improve training.

## Approach
Due to CPU limitations, I decided to use a lightweight simulator and chose the [MuJoCo](https://gymnasium.farama.org/environments/mujoco/) environment in Gynasium. I trained a 2D walker, `walker2d-v4`,using Proximal Policy Optimization (PPO), a policy-gradient RL algorithm that works by updating a neural network policy using sampled rollouts.

I first trained it on the default reward policy, which led to early falls because the reward primarily incentivized speed and not stability. I then introduced a [custom reward policy](CustomReward.py), rewarding low joint torques and uprightedness. This helped the model learn more long-term movements, develop a consistent gait, and survive longer without falling.

### CustomReward.py
The custom reward works by taking the default environment reward and subtracting weighted penalties for torso angle, knee angle, and left-right joint asymmetry. Since these penalties lower the cumulative return, PPO is discouraged from unstable or aggressive motions, instead learning actions that achieve desired behavior like upright torso and balanced joint movements. This ***reward shaping*** biases the policy toward smoother and stable walking behaviors.

## Project Structure
```
walker2d-demo/
├── assets/
|   ├── default_policy_video.gif
|   └── custom_policy_video.gif
├── plots/
|   ├── learning_curve_custom.csv
|   ├── learning_curve_custom.png
|   ├── learning_curve_default.csv
|   ├── learning_curve_default.png
|   └── plot_learning_curves.ipynb 
├── CustomReward.py
├── .gitignore
├── README.md
├── record_video.py
├── requirements.txt 
├── train_walker2d.py
```

## Results
### Default Policy
| Learning Curve | Trained Policy Rollout |
|---------------|------------------------|
| <img src="plots/learning_curve_default.png" height="250"> | <img src="assets/default_policy_video.gif" height="250"> |
| *Episode reward over training with the default reward.* | *Behavior under default policy, showing early instability.* |

---

### Custom Policy
| Learning Curve | Trained Policy Rollout |
|---------------|------------------------|
| <img src="plots/learning_curve_custom.png" height="250"> | <img src="assets/custom_policy_video.gif" height="250"> |
| *Episode reward over training with the custom reward.* | *Behavior with custom reward, showing improved stability & symmetry.* |