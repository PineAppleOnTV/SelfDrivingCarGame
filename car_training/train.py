import os
import torch

from policy import PolicyNetwork
from car_env import CarEnv, MultiCarEnv
from rollout import collect_parallel_rollout
from reinforce import update_policy
from settings import NUM_CARS

NUM_UPDATES = 2000
STEPS_PER_ROLLOUT = 500
RENDER_EVERY = 20
SAVE_EVERY = 20
CHECKPOINT_PATH = "car_policy.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

policy = PolicyNetwork(obs_dim=6, action_dim=2, device=device)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

start_update = 0
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_update = checkpoint["update"] + 1
    print(f"Resumed from update {start_update}")

def save_checkpoint(update):
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "update": update,
    }, CHECKPOINT_PATH)

watch_env = CarEnv(render=True)       # real window, constructed first
train_env = MultiCarEnv(NUM_CARS)     # headless, many cars, reuses same display

for update in range(start_update, start_update + NUM_UPDATES):
    log_probs, normalized_returns, stats = collect_parallel_rollout(
        train_env, policy, STEPS_PER_ROLLOUT, gamma=0.99
    )

    loss = update_policy(policy, optimizer, log_probs, normalized_returns)

    print(f"Update {update:4d} | episodes: {stats['num_episodes']:4d} | "
          f"avg_length: {stats['avg_length']:6.1f} | avg_reward: {stats['avg_reward']:7.3f} | loss: {loss:8.3f}")

    if (update + 1) % SAVE_EVERY == 0:
        save_checkpoint(update)

    if (update + 1) % RENDER_EVERY == 0:
        obs = watch_env.reset()
        done = False
        steps = 0
        while not done and steps < 3000:
            action, _ = policy.get_action(obs)
            obs, reward, done, info = watch_env.step(action)
            watch_env.render()
            steps += 1

save_checkpoint(start_update + NUM_UPDATES - 1)
