# watch_only.py

import torch

from policy import PolicyNetwork
from car_env import CarEnv
from settings import SENSOR_COUNT

CHECKPOINT_PATH = "car_policy.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(f"Using device: {device}")

policy = PolicyNetwork(obs_dim=6, action_dim=2, device=device)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
policy.load_state_dict(checkpoint["policy_state_dict"])
print(f"Loaded policy from update {checkpoint['update']}")

env = CarEnv(render=True)
obs = env.reset()

while True:
    action, _ = policy.get_action(obs)
    obs, reward, done, info = env.step(action)
    env.render()

    sensor_readings = obs[:SENSOR_COUNT]
    speed_reading = obs[SENSOR_COUNT]
    #print(f"sensors: {sensor_readings} | speed: {speed_reading:.3f}")

    if done:
        obs = env.reset()