# test_update.py

import torch

from policy import PolicyNetwork
from car_env import CarEnv
from rollout import collect_episode
from returns import compute_returns, normalize_returns
from reinforce import update_policy

policy = PolicyNetwork(obs_dim=6, action_dim=2)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
env = CarEnv()

observations, actions, log_probs, rewards = collect_episode(env, policy)

returns = compute_returns(rewards, gamma=0.99)
normalized = normalize_returns(returns)

loss = update_policy(policy, optimizer, log_probs, normalized)

print(f"Episode length: {len(rewards)}")
print(f"Loss: {loss}")