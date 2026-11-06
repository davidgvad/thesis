"""Predict response labels from a frozen ego's decision representations.

Inputs: representations [N,128] and integer labels [N]. Use the GRU hidden
state or Transformer memory['representation'] at the selected decision.
Keep source episodes separate between training and testing.
"""

import torch
from torch import nn
from torch.nn import functional as F


class LinearProbe(nn.Module):
    def __init__(self, num_classes, input_size=128):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, representations):
        return self.linear(representations)


class NonlinearProbe(nn.Module):
    def __init__(self, num_classes, input_size=128, hidden_size=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, representations):
        return self.layers(representations)


def fit_probe(
    probe, representations, labels, epochs=10, batch_size=128,
    learning_rate=1e-3, weight_decay=1e-4,
):
    """Train only the probe; return the mean training loss for each epoch."""
    device = next(probe.parameters()).device
    representations = representations.detach().to(device)
    labels = labels.to(device)
    optimizer = torch.optim.Adam(
        probe.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    examples = labels.numel()
    losses = []
    probe.train()
    for _ in range(epochs):
        order = torch.randperm(examples, device=device)
        total_loss = 0.0
        for indices in order.split(batch_size):
            logits = probe(representations[indices])
            loss = F.cross_entropy(logits, labels[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * indices.numel()
        losses.append(total_loss / examples)
    return losses


@torch.no_grad()
def evaluate_probe(probe, representations, labels):
    """Return accuracy on representations from separate test episodes."""
    probe.eval()
    device = next(probe.parameters()).device
    logits = probe(representations.to(device))
    predictions = logits.argmax(dim=-1)
    return (predictions == labels.to(device)).float().mean().item()
