"""Supervised diagnostic: predict the appropriate response from legal history.

This is independent of the ego policy and PPO. Each example starts at an episode
boundary and ends at a decision observation, before its action/reward. Never use
critic observations, hidden state labels, or future outcomes as input features.

Run this file for a toy check, or pass --data-file with labelled Overcooked data.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from models import CNNEncoder
from toy_pomdp import MemoryPOMDP, WAIT


class HistoryClassifier(nn.Module):
    """[T,B,*obs_shape] and lengths [B] -> response logits [B,num_classes].

    lengths includes the decision observation. Entries at/after lengths are
    excluded. Supply one episode per sequence, without crossing reset boundaries.
    current_only=True uses just the last legal observation, without actions/rewards.
    """

    def __init__(
        self, obs_shape, num_classes, num_actions=6, max_steps=400, current_only=False,
    ):
        super().__init__()
        if num_classes < 2 or num_actions < 1 or max_steps < 1:
            raise ValueError("Use at least two classes, one action, and one timestep.")
        self.obs_shape = tuple(obs_shape)
        self.num_actions = num_actions
        self.max_steps = max_steps
        self.current_only = current_only
        if len(self.obs_shape) == 1:
            self.encoder = nn.Sequential(
                nn.Linear(self.obs_shape[0], 128), nn.ReLU(), nn.LayerNorm(128),
            )
        elif len(self.obs_shape) == 3:
            self.encoder = CNNEncoder(self.obs_shape)  # Fresh weights, not the ego's CNN.
        else:
            raise ValueError("obs_shape must describe a vector [F] or a grid [H,W,C].")
        self.head = nn.Linear(128, num_classes)
        if not current_only:
            self.input_layer = nn.Linear(128 + num_actions + 1, 128)
            layer = nn.TransformerEncoderLayer(
                d_model=128, nhead=4, dim_feedforward=128, dropout=0.0,
                activation="relu", batch_first=True, norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                layer, num_layers=1, norm=nn.LayerNorm(128), enable_nested_tensor=False,
            )
            # Fixed sinusoidal positions; no learned lookup tied to an episode length.
            times = torch.arange(max_steps)[:, None]
            frequencies = torch.exp(-math.log(10000) * torch.arange(0, 128, 2) / 128)
            angles = times * frequencies
            positions = torch.stack([angles.sin(), angles.cos()], dim=-1).flatten(-2)
            self.register_buffer("positions", positions, persistent=False)

    def forward(self, obs, previous_action, previous_reward, lengths):
        if tuple(obs.shape[2:]) != self.obs_shape or obs.ndim != len(self.obs_shape) + 2:
            raise ValueError("obs must have shape [T,B,*obs_shape].")
        steps, batch_size = obs.shape[:2]
        if steps < 1 or steps > self.max_steps:
            raise ValueError("History length must be between 1 and max_steps; do not silently truncate.")
        if lengths.shape != (batch_size,) or lengths.dtype != torch.long:
            raise ValueError("lengths must be a torch.long tensor shaped [B].")
        if not ((lengths >= 1) & (lengths <= steps)).all().item():
            raise ValueError("Every length must identify a nonempty legal prefix within obs.")
        batch = torch.arange(batch_size, device=obs.device)
        if self.current_only:
            # Observation alone: even the previous reward/action is deliberately omitted.
            return self.head(self.encoder(obs[lengths - 1, batch].float()))
        if previous_action.shape != (steps, batch_size) or previous_reward.shape != (steps, batch_size):
            raise ValueError("Previous actions and rewards must have shape [T,B].")

        valid = torch.arange(steps, device=obs.device)[:, None] < lengths[None]
        # Clean excluded data BEFORE encoding (including NaNs or invalid padded actions).
        obs_mask = valid.reshape(steps, batch_size, *([1] * len(self.obs_shape)))
        safe_obs = obs.masked_fill(~obs_mask, 0)
        actions = previous_action.masked_fill(~valid, 0)
        rewards = previous_reward.masked_fill(~valid, 0).float()
        actions[0] = 0  # Also sanitize invalid placeholders at the episode boundary.
        action_features = F.one_hot(actions.long(), self.num_actions).float()
        action_features[0] = 0  # No preceding episode's action/reward at the start.
        rewards = rewards.clone()
        rewards[0] = 0
        tokens = self.input_layer(torch.cat([
            self.encoder(safe_obs.float()), action_features, rewards[..., None],
        ], dim=-1)).transpose(0, 1)
        tokens = tokens + self.positions[:steps]
        future = torch.ones((steps, steps), dtype=torch.bool, device=obs.device).triu(1)
        outputs = self.transformer(tokens, mask=future, src_key_padding_mask=~valid.T)
        return self.head(outputs[batch, lengths - 1])


def collect_toy_data(episodes=2048, seed=0, clue_timing="early", clue_accuracy=1.0, delay_steps=5):
    """One legal decision prefix per episode. The label is never an input feature."""
    env = MemoryPOMDP(
        batch_size=episodes, seed=seed, clue_timing=clue_timing,
        clue_accuracy=clue_accuracy, delay_steps=delay_steps,
    )
    state = env.reset()
    observations = []
    for _ in range(env.max_steps):
        observations.append(env.get_obs(state))
        if state.time == env.decision_step:
            break  # Stop BEFORE choosing or observing the final reward / late clue.
        state, _, _, _ = env.step(state, torch.full_like(state.condition, WAIT))
    obs = torch.stack(observations)
    return {
        "obs": obs,
        "previous_action": torch.full(obs.shape[:2], WAIT, dtype=torch.long),
        "previous_reward": torch.zeros(obs.shape[:2]),
        "lengths": torch.full((episodes,), obs.shape[0], dtype=torch.long),
        "labels": state.condition.clone(),
        "episode_ids": torch.arange(episodes),
    }


def validate_data(data):
    """Validate the file format; the caller must ensure features are legal history."""
    required = ("obs", "previous_action", "previous_reward", "lengths", "labels", "episode_ids")
    if not all(key in data and isinstance(data[key], torch.Tensor) for key in required):
        raise ValueError(f"Data must contain tensors: {', '.join(required)}.")
    if data["obs"].ndim not in (3, 5):
        raise ValueError("obs must be [T,N,F] or [T,N,H,W,C].")
    steps, examples = data["obs"].shape[:2]
    if steps < 1 or examples < 1:
        raise ValueError("Provide at least one example and one timestep.")
    for name in ("previous_action", "previous_reward"):
        if data[name].shape != (steps, examples):
            raise ValueError(f"{name} must have shape [T,N].")
    if data["previous_action"].dtype != torch.long:
        raise ValueError("previous_action must be torch.long.")
    for name in ("lengths", "labels", "episode_ids"):
        if data[name].shape != (examples,) or data[name].dtype != torch.long:
            raise ValueError(f"{name} must be torch.long shaped [N].")
    if not ((data["lengths"] >= 1) & (data["lengths"] <= steps)).all().item():
        raise ValueError("lengths must be between 1 and T.")
    if (data["labels"] < 0).any().item():
        raise ValueError("Class labels must be nonnegative integers starting at zero.")


def select_examples(data, indices):
    sequence_fields = ("obs", "previous_action", "previous_reward")
    fields = sequence_fields + ("lengths", "labels", "episode_ids")
    return {key: data[key][:, indices] if key in sequence_fields else data[key][indices]
            for key in fields}


def split_by_episode(data, seed=0, train_fraction=0.8):
    """Keep every example from the same source episode on the same side of the split."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    ids = torch.unique(data["episode_ids"])
    if ids.numel() < 2:
        raise ValueError("Need at least two independent source episodes for a split.")
    generator = torch.Generator().manual_seed(seed)
    ids = ids[torch.randperm(ids.numel(), generator=generator)]
    count = max(1, min(ids.numel() - 1, int(ids.numel() * train_fraction)))
    train = torch.isin(data["episode_ids"], ids[:count])
    return select_examples(data, train), select_examples(data, ~train)


def model_batch(data, indices, device):
    batch = select_examples(data, indices)
    return {key: value.to(device) for key, value in batch.items() if key != "episode_ids"}


def fit_classifier(model, data, epochs=10, batch_size=128, learning_rate=1e-3, seed=0):
    """Supervised cross-entropy training. No ego parameters or PPO updates involved."""
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive.")
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    examples = data["labels"].numel()
    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        order = torch.randperm(examples, generator=generator)
        total_loss = 0.0
        for indices in order.split(batch_size):
            batch = model_batch(data, indices, device)
            logits = model(batch["obs"], batch["previous_action"], batch["previous_reward"], batch["lengths"])
            loss = F.cross_entropy(logits, batch["labels"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * indices.numel()
        history.append({"epoch": epoch, "train_loss": total_loss / examples})
    return history


@torch.no_grad()
def evaluate_classifier(model, data, batch_size=128):
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    correct = total_loss = 0.0
    examples = data["labels"].numel()
    class_counts = [0] * model.head.out_features
    class_correct = [0] * model.head.out_features
    try:
        for indices in torch.arange(examples).split(batch_size):
            batch = model_batch(data, indices, device)
            logits = model(batch["obs"], batch["previous_action"], batch["previous_reward"], batch["lengths"])
            total_loss += F.cross_entropy(logits, batch["labels"], reduction="sum").item()
            matches = logits.argmax(dim=-1) == batch["labels"]
            correct += matches.sum().item()
            for label in range(model.head.out_features):
                selected = batch["labels"] == label
                class_counts[label] += selected.sum().item()
                class_correct[label] += (matches & selected).sum().item()
    finally:
        model.train(was_training)
    per_class = {str(label): class_correct[label] / count if count else None
                 for label, count in enumerate(class_counts)}
    return {
        "accuracy": correct / examples, "loss": total_loss / examples,
        "examples": examples, "accuracy_per_class": per_class,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, help="Optional .pt dictionary of labelled legal histories.")
    parser.add_argument("--episodes", type=int, default=2048, help="Toy examples if no data file is given.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-actions", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--clue-timing", choices=("early", "late"), default="early")
    parser.add_argument("--clue-accuracy", type=float, default=1.0)
    parser.add_argument("--delay-steps", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs/history_classifier"))
    args = parser.parse_args()
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be positive.")
    torch.set_num_threads(args.cpu_threads)
    data = torch.load(args.data_file, map_location="cpu", weights_only=True) if args.data_file else collect_toy_data(
        args.episodes, args.seed, args.clue_timing, args.clue_accuracy, args.delay_steps,
    )
    train, test = split_by_episode(data, seed=args.seed)
    num_classes = int(data["labels"].max().item()) + 1
    if num_classes < 2:
        parser.error("Need at least two response classes.")
    # A fixed training schedule; held-out test data is evaluated only afterward.
    args.output.mkdir(parents=True, exist_ok=False)
    config = {
        "obs_shape": list(data["obs"].shape[2:]), "num_classes": num_classes,
        "num_actions": args.num_actions, "max_steps": args.max_steps,
    }
    results = {}
    for name, current_only in (("current_observation", True), ("history", False)):
        torch.manual_seed(args.seed)
        model = HistoryClassifier(**config, current_only=current_only).to(args.device)
        history = fit_classifier(model, train, args.epochs, args.batch_size, args.learning_rate, args.seed)
        result = evaluate_classifier(model, test, args.batch_size)
        results[name] = result
        torch.save({"model": model.state_dict(), "config": {**config, "current_only": current_only}},
                   args.output / f"{name}.pt")
        with (args.output / f"{name}_training.json").open("w") as file:
            json.dump(history, file, indent=2)
        print(f"{name:20s} | Held-out accuracy {result['accuracy']:.3f} | Episodes/examples {result['examples']}", flush=True)
    majority = train["labels"].bincount(minlength=num_classes).argmax()
    results["majority_baseline_accuracy"] = (test["labels"] == majority).float().mean().item()
    payload = {"settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
               "train_examples": train["labels"].numel(), "test_examples": test["labels"].numel(),
               "train_episodes": train["episode_ids"].unique().numel(),
               "test_episodes": test["episode_ids"].unique().numel(),
               "results": results}
    with (args.output / "results.json").open("w") as file:
        json.dump(payload, file, indent=2)
    print(f"Results saved to {args.output / 'results.json'}")


if __name__ == "__main__":
    main()
