import numpy as np


def compute_returns(rewards, gamma=0.99):
    """
    Converts a list of per-step rewards into discounted returns:
    for each timestep, "how much total (discounted) reward followed
    from here onward."

    Computed backward through the episode, since G_t = r_t + gamma * G_(t+1)
    lets each return reuse the one after it, instead of resumming from
    scratch for every timestep.
    """
    returns = [0.0] * len(rewards)
    running_return = 0.0

    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return

    return returns


def normalize_returns(returns):
    """
    Rescales returns to mean 0, std 1. Keeps the learning signal at a
    consistent scale across episodes of very different lengths/outcomes,
    which policy gradient updates are sensitive to.
    """
    returns = np.array(returns, dtype=np.float32)
    mean = returns.mean()
    std = returns.std() + 1e-8  # avoid divide-by-zero if std happens to be 0
    return (returns - mean) / std
