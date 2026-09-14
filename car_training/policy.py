import torch
import torch.nn as nn
from torch.distributions import Normal

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size=32, device=None):
        super().__init__()
        self.device = device or torch.device("cpu")

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.to(self.device)  # moves all layers + log_std to GPU/CPU as one call

    def forward(self, obs):
        features = self.shared(obs)
        mean = self.mean_head(features)
        std = torch.exp(self.log_std)
        return Normal(mean, std)

    def get_action(self, obs):
        """Single-observation version - still used by watch_only.py / single CarEnv."""
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        distribution = self.forward(obs_tensor)
        action = distribution.sample()
        log_prob = distribution.log_prob(action).sum(dim=-1)
        action_clipped = torch.clamp(action, -1.0, 1.0)
        return action_clipped.squeeze(0).cpu().numpy(), log_prob.squeeze(0)

    def get_actions_batch(self, obs_batch):
        """
        Batched version - runs ALL cars' observations through the network
        in a single forward pass. This is the actual source of any speedup,
        on CPU or GPU alike - one matmul over 100 rows beats 100 separate calls.
        """
        obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=self.device)
        distribution = self.forward(obs_tensor)
        actions = distribution.sample()
        log_probs = distribution.log_prob(actions).sum(dim=-1)
        actions_clipped = torch.clamp(actions, -1.0, 1.0)
        return actions_clipped.cpu().numpy(), log_probs
