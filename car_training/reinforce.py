import torch


def update_policy(policy, optimizer, log_probs, normalized_returns):
    log_probs_tensor = torch.stack(log_probs)
    returns_tensor = torch.as_tensor(
        normalized_returns, dtype=torch.float32, device=log_probs_tensor.device
    )

    loss = -(log_probs_tensor * returns_tensor).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()