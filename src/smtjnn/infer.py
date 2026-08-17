"""Inference harness: T stochastic passes per image, logits averaged."""

import torch


@torch.no_grad()
def evaluate(model, loader, source, T: int = 16, device="cpu") -> float:
    """Classification accuracy with T-pass logit averaging.

    Device state persists across the T passes for one batch (same physical
    devices queried repeatedly) and is reset between batches (independent
    device initial conditions per image group).
    """
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        source.reset_batch()
        logits = torch.zeros(x.shape[0], model.dims[-1], device=device)
        for _ in range(T):
            logits += model(x, source=source)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total
