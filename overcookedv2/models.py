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


def run_gru(gru, inputs, hidden, episode_start):
    """Run one timestep or an ordered sequence; reset finished kitchens."""
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

    def __init__(self, obs_shape):
        super().__init__()
        self.encoder = CNNEncoder(obs_shape)
        self.input_layer = nn.Sequential(nn.Linear(135, 128), nn.ReLU())
        # GRUCell is one GRU layer, applied timestep by timestep.
        self.gru = nn.GRUCell(128, 128)
        self.action_head = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 6)
        )

    def forward(
        self, obs, previous_action, previous_reward, hidden, episode_start,
        reset_memory=False,
    ):
        features = self.encoder(obs)
        action_features = F.one_hot(previous_action.long(), 6).float()
        action_features = action_features.masked_fill(episode_start[..., None], 0)
        reward_feature = previous_reward.masked_fill(episode_start, 0)[..., None]
        inputs = torch.cat([features, action_features, reward_feature], dim=-1)
        inputs = self.input_layer(inputs)

        # Optional evaluation ablation: forget memory at every timestep.
        memory_resets = torch.ones_like(episode_start) if reset_memory else episode_start
        outputs, hidden = run_gru(self.gru, inputs, hidden, memory_resets)
        return self.action_head(outputs), hidden


class CentralCritic(nn.Module):
    """Independent network: full kitchen input -> one team value per kitchen."""

    def __init__(self, obs_shape):
        super().__init__()
        self.encoder = CNNEncoder(obs_shape)
        self.gru = nn.GRUCell(128, 128)
        self.value_head = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def forward(self, obs, hidden, episode_start):
        inputs = self.encoder(obs)
        outputs, hidden = run_gru(self.gru, inputs, hidden, episode_start)
        return self.value_head(outputs).squeeze(-1), hidden
