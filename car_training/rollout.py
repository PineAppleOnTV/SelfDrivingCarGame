# rollout.py

from returns import compute_returns, normalize_returns

def collect_parallel_rollout(env, policy, num_steps, gamma=0.99):
    """
    Runs num_steps timesteps across all cars simultaneously. Each car's
    episode is finalized independently the moment it goes off-track (or
    when the rollout ends, as a truncated episode) - since cars reset
    and keep going within the same fixed-length rollout.
    """
    num_cars = env.num_cars
    obs = env.reset()

    car_log_probs = [[] for _ in range(num_cars)]
    car_rewards = [[] for _ in range(num_cars)]

    finished_log_probs = []
    finished_returns = []
    episode_lengths = []
    episode_rewards = []

    for _ in range(num_steps):
        actions, log_probs = policy.get_actions_batch(obs)
        next_obs, rewards, dones, infos = env.step(actions)

        for i in range(num_cars):
            car_log_probs[i].append(log_probs[i])
            car_rewards[i].append(rewards[i])

            if dones[i]:
                returns = compute_returns(car_rewards[i], gamma)
                finished_log_probs.extend(car_log_probs[i])
                finished_returns.extend(returns)
                episode_lengths.append(len(car_rewards[i]))
                episode_rewards.append(sum(car_rewards[i]))
                car_log_probs[i] = []
                car_rewards[i] = []

        obs = next_obs

    # finalize any still-in-progress (truncated) episodes at rollout end
    for i in range(num_cars):
        if car_rewards[i]:
            returns = compute_returns(car_rewards[i], gamma)
            finished_log_probs.extend(car_log_probs[i])
            finished_returns.extend(returns)
            episode_lengths.append(len(car_rewards[i]))
            episode_rewards.append(sum(car_rewards[i]))

    normalized_returns = normalize_returns(finished_returns)

    stats = {
        "num_episodes": len(episode_lengths),
        "avg_length": sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0,
        "avg_reward": sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0,
    }

    return finished_log_probs, normalized_returns, stats
