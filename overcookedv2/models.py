import math

import torch
from torch import nn
from torch.nn import functional as F


class CNNEncoder(nn.Module):
    """An observation grid becomes 128 features."""

    def __init__(self, obs_shape):
        super().__init__()
        height, width, channels = obs_shape
        self.layers = nn.Sequential(
            nn.Conv2d(channels, 128, 1), nn.ReLU(),
            nn.Conv2d(128, 128, 1), nn.ReLU(),
            nn.Conv2d(128, 8, 1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * height * width, 128), nn.ReLU(),
            nn.LayerNorm(128),
        )

    def forward(self, obs):
        # Accept [B, H, W, C] or [T, B, H, W, C].
        batch_shape = obs.shape[:-3]
        height, width, channels = obs.shape[-3:]
        grids = obs.reshape(-1, height, width, channels)
        grids = grids.permute(0, 3, 1, 2).contiguous()
        features = self.layers(grids.float())
        return features.reshape(*batch_shape, 128)


def make_encoder(obs_shape):
    """Vectors use a small MLP; grids use the kitchen CNN. Both output 128 features."""
    if len(obs_shape) == 1:
        return nn.Sequential(
            nn.Linear(obs_shape[0], 128), nn.ReLU(), nn.LayerNorm(128),
        )
    return CNNEncoder(obs_shape)


def run_gru(gru, inputs, hidden, episode_start):
    """Run one timestep or an ordered sequence, reset finished kitchens."""
    single_step = inputs.ndim == 2
    if single_step:
        inputs = inputs[None]
        episode_start = episode_start[None]

    if hidden is None:
        hidden = inputs.new_zeros((inputs.shape[1], 128))

    outputs = []
    for t in range(inputs.shape[0]):
        hidden = hidden.masked_fill(episode_start[t, :, None], 0)
        hidden = gru(inputs[t], hidden)
        outputs.append(hidden)

    outputs = torch.stack(outputs)
    if single_step:
        outputs = outputs[0]
    return outputs, hidden


class GRUActor(nn.Module):
    """Uses only the ego observation, previous action/reward, and its memory."""

    def __init__(self, obs_shape, num_actions=6):
        super().__init__()
        self.num_actions = num_actions
        self.encoder = make_encoder(obs_shape)
        self.input_layer = nn.Sequential(nn.Linear(128 + num_actions + 1, 128), nn.ReLU())
        # GRUCell is one GRU layer, applied timestep by timestep.
        self.gru = nn.GRUCell(128, 128)
        self.action_head = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, num_actions)
        )

    def forward(
        self, obs, previous_action, previous_reward, hidden, episode_start,
        reset_memory=False,
    ):
        features = self.encoder(obs)
        action_features = F.one_hot(previous_action.long(), self.num_actions).float()
        action_features = action_features.masked_fill(episode_start[..., None], 0)
        reward_feature = previous_reward.masked_fill(episode_start, 0)[..., None]
        inputs = torch.cat([features, action_features, reward_feature], dim=-1)
        inputs = self.input_layer(inputs)

        # Optional evaluation ablation: forget memory at every timestep.
        memory_resets = torch.ones_like(episode_start) if reset_memory else episode_start
        outputs, hidden = run_gru(self.gru, inputs, hidden, memory_resets)
        return self.action_head(outputs), hidden


class TransformerActor(nn.Module):
    """Same actor interface as GRUActor, with a dictionary of recent inputs.

    One timestep is one token: CNN(observation), previous action, previous reward.
    Memory keeps raw inputs so PPO can re-encode history with the current weights.
    During collection, cached tokens avoid running the CNN on old observations.
    """

    def __init__(self, obs_shape, context_length=400):
        super().__init__()
        if context_length < 1:
            raise ValueError("context_length must be positive.")
        self.context_length = context_length
        self.encoder = CNNEncoder(obs_shape)
        self.input_layer = nn.Sequential(nn.Linear(135, 128), nn.ReLU())

        # One pre-normalized Transformer block. Four heads each use 32 features.
        self.attention_norm = nn.LayerNorm(128)
        self.attention = nn.MultiheadAttention(128, 4, dropout=0.0, batch_first=True)
        self.feed_forward_norm = nn.LayerNorm(128)
        self.feed_forward = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128),
        )
        self.output_norm = nn.LayerNorm(128)
        self.action_head = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 6),
        )
        # Fixed sinusoidal positions add no learned parameters.
        frequencies = torch.exp(-math.log(10000) * torch.arange(0, 128, 2) / 128)
        self.register_buffer("frequencies", frequencies, persistent=False)

    def _encode_tokens(self, obs, previous_action, previous_reward, episode_start):
        features = self.encoder(obs)
        actions = F.one_hot(previous_action.long(), 6).float()
        actions = actions.masked_fill(episode_start[..., None], 0)
        rewards = previous_reward.masked_fill(episode_start, 0)[..., None]
        return self.input_layer(torch.cat([features, actions, rewards], dim=-1))

    def forward(
        self, obs, previous_action, previous_reward, hidden, episode_start,
        reset_memory=False,
    ):
        # The public interface uses [T,B,...]; attention uses [B,T,...].
        single_step = obs.ndim == 4
        if single_step:
            obs = obs[None]
            previous_action = previous_action[None]
            previous_reward = previous_reward[None]
            episode_start = episode_start[None]
        obs = obs.transpose(0, 1)
        previous_action = previous_action.transpose(0, 1)
        previous_reward = previous_reward.transpose(0, 1)
        episode_start = episode_start.transpose(0, 1)
        batch_size, steps = previous_action.shape

        current = {
            "obs": obs, "previous_action": previous_action,
            "previous_reward": previous_reward, "episode_start": episode_start,
        }
        prefix_length = 0 if hidden is None else hidden["obs"].shape[1]
        history = current if hidden is None else {
            name: torch.cat([hidden[name], value], dim=1)
            for name, value in current.items()
        }

        # Positions count from zero at each episode start, including when a
        # rollout continues an episode that began in an earlier collection.
        times = torch.arange(1, steps + 1, device=obs.device)[None]
        resets = torch.ones_like(episode_start) if reset_memory else episode_start
        last_reset = torch.where(resets, times, 0).cummax(dim=1).values
        first_position = torch.zeros(batch_size, device=obs.device, dtype=torch.long)
        if hidden is not None:
            first_position = hidden["next_position"]
        positions = torch.where(last_reset > 0, times - last_reset,
                                first_position[:, None] + times - 1)
        if hidden is not None:
            positions = torch.cat([hidden["positions"], positions], dim=1)

        # PPO always recomputes the raw history; collection reuses cached tokens.
        if not torch.is_grad_enabled() and hidden is not None and "tokens" in hidden:
            tokens = torch.cat([hidden["tokens"], self._encode_tokens(**current)], dim=1)
        else:
            tokens = self._encode_tokens(**history)
        angles = positions[..., None] * self.frequencies
        position_features = torch.stack([angles.sin(), angles.cos()], dim=-1).flatten(-2)
        inputs = tokens + position_features

        # Each query may read only the same episode and at most context_length
        # tokens, including itself. No future observations or other episodes.
        boundaries = history["episode_start"]
        if reset_memory:
            boundaries = torch.ones_like(boundaries)
        episodes = boundaries.long().cumsum(dim=1)
        length = inputs.shape[1]
        keys = torch.arange(length, device=obs.device)
        queries = keys[prefix_length:]
        age = queries[:, None] - keys[None, :]
        allowed = (age >= 0) & (age < self.context_length)
        allowed = allowed[None] & (episodes[:, prefix_length:, None] == episodes[:, None, :])
        # MultiheadAttention expects True for blocked entries, one mask per head.
        blocked = (~allowed).repeat_interleave(self.attention.num_heads, dim=0)

        normalized = self.attention_norm(inputs)
        attended, _ = self.attention(
            normalized[:, prefix_length:], normalized, normalized,
            attn_mask=blocked, need_weights=False,
        )
        # Only current queries need outputs. With one block, old keys/values
        # depend on their input tokens, so collection need not recompute old outputs.
        outputs = inputs[:, prefix_length:] + attended
        outputs = outputs + self.feed_forward(self.feed_forward_norm(outputs))
        outputs = self.output_norm(outputs)
        logits = self.action_head(outputs).transpose(0, 1)

        # Save at most C-1 earlier tokens: the next observation is the Cth token.
        # These are detached inputs, not a computation graph across PPO updates.
        start = max(0, length - (self.context_length - 1))
        memory = {name: value[:, start:].detach() for name, value in history.items()}
        memory["positions"] = positions[:, start:].detach()
        memory["next_position"] = positions[:, -1].detach() + 1
        memory["tokens"] = tokens[:, start:].detach()
        memory["representation"] = outputs[:, -1].detach()  # Small [B,128] trace.
        return (logits[0] if single_step else logits), memory

class CentralCritic(nn.Module):
    """Independent GRU value network for grid or vector observations."""

    def __init__(self, obs_shape):
        super().__init__()
        self.encoder = make_encoder(obs_shape)
        self.gru = nn.GRUCell(128, 128)
        self.value_head = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def forward(self, obs, hidden, episode_start):
        inputs = self.encoder(obs)
        outputs, hidden = run_gru(self.gru, inputs, hidden, episode_start)
        return self.value_head(outputs).squeeze(-1), hidden
