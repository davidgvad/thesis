from dataclasses import dataclass

import torch
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_

from commons import StaticObject as S


@dataclass
class PPOConfig:
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.25
    epochs: int = 4
    minibatch_size: int = 4  # Kitchens/episode sequences, not timesteps.


def get_critic_obs(env, state, ego_index=0):
    """Full grid, actual recipe, pot/button timers, and remaining episode time."""
    obs = env.get_obs(state, full_grid=True)[:, ego_index].clone()
    ingredients = env.num_ingredients

    # Skip SELF, OTHERS, static maps, ingredient piles, and grid contents.
    recipe_start = (
        2 * (7 + ingredients) + 6 + ingredients + (2 + ingredients)
    )
    recipe_end = recipe_start + 2 + ingredients
    obs[..., recipe_start:recipe_end] = env._decode_items(state.recipe)[:, None, None]

    static = state.grid[..., 0]
    has_timer = (static == int(S.POT)) | (static == int(S.BUTTON_RECIPE_INDICATOR))
    obs[..., recipe_end] = torch.where(has_timer, state.grid[..., 2], 0)

    remaining = (env.max_steps - state.time).float().clamp_min(0) / env.max_steps
    time_map = remaining[:, None, None, None].expand(
        -1, state.grid.shape[1], state.grid.shape[2], 1
    )
    return torch.cat([obs, time_map], dim=-1)


@torch.no_grad()
def collect_rollout(
    env, actor, critic, teammate, ego_index=0, steps=None, reset_actor_memory=False,
):
    """Collect ordered episodes. Defaults to the entire finite episode horizon.

    teammate(env, state, teammate_index) returns fixed actions shaped [B].
    Each collection starts fresh, with zero action/reward history and memory.
    A shorter collection bootstraps its unfinished boundary; it is a truncated
    training sequence. Use the default full horizon for the pilot.
    """
    if env.layout.agent_positions.shape[1] != 2 or ego_index not in (0, 1):
        raise ValueError("This pilot uses two agents: one ego and one fixed teammate.")
    steps = env.max_steps if steps is None else steps
    if steps < 1 or steps > env.max_steps:
        raise ValueError("steps must be between 1 and the episode horizon.")

    state = env.reset()
    obs = env.get_obs(state)
    batch_size = state.grid.shape[0]
    device = state.grid.device
    previous_action = torch.zeros(batch_size, dtype=torch.long, device=device)
    previous_reward = torch.zeros(batch_size, device=device)
    episode_start = torch.ones(batch_size, dtype=torch.bool, device=device)
    actor_hidden = None
    critic_hidden = None
    data = {key: [] for key in (
        "obs", "critic_obs", "previous_action", "previous_reward", "episode_start",
        "actions", "joint_actions", "log_prob", "values", "rewards", "done",
        "actor_hidden", "action_probs",
    )}

    for _ in range(steps):
        ego_obs = obs[:, ego_index]
        critic_obs = get_critic_obs(env, state, ego_index)
        logits, actor_hidden = actor(
            ego_obs, previous_action, previous_reward, actor_hidden, episode_start,
            reset_memory=reset_actor_memory,
        )
        values, critic_hidden = critic(critic_obs, critic_hidden, episode_start)
        distribution = Categorical(logits=logits)
        action = distribution.sample()

        joint_actions = torch.empty((batch_size, 2), dtype=torch.long, device=device)
        joint_actions[:, ego_index] = action
        joint_actions[:, 1 - ego_index] = teammate(env, state, 1 - ego_index)
        next_state, next_obs, reward, _, done = env.step(state, joint_actions)
        # Shaped rewards are deliberately excluded from the training objective.

        transition = {
            "obs": ego_obs, "critic_obs": critic_obs,
            "previous_action": previous_action, "previous_reward": previous_reward,
            "episode_start": episode_start, "actions": action,
            "joint_actions": joint_actions, "log_prob": distribution.log_prob(action),
            "values": values, "rewards": reward, "done": done,
            "actor_hidden": actor_hidden, "action_probs": distribution.probs,
        }
        for key, value in transition.items():
            data[key].append(value)

        state, obs = next_state, next_obs
        episode_start = done
        previous_action = action.masked_fill(done, 0)
        previous_reward = reward.masked_fill(done, 0)
        actor_hidden = actor_hidden.masked_fill(done[:, None], 0)
        critic_hidden = critic_hidden.masked_fill(done[:, None], 0)

    next_value, _ = critic(
        get_critic_obs(env, state, ego_index), critic_hidden, episode_start
    )
    rollout = {key: torch.stack(values) for key, values in data.items()}
    # A reset kitchen's value belongs to a different episode: never bootstrap it.
    rollout["next_value"] = next_value.masked_fill(episode_start, 0)
    rollout["reset_actor_memory"] = reset_actor_memory
    return rollout


@torch.no_grad()
def compute_gae(rewards, values, done, next_value, gamma=0.99, gae_lambda=0.95):
    """Advantages and value targets, both [T, B], with no gradients."""
    advantages = torch.zeros_like(rewards)
    advantage = torch.zeros_like(next_value)
    for t in reversed(range(rewards.shape[0])):
        continues = (~done[t]).float()
        delta = rewards[t] + gamma * continues * next_value - values[t]
        advantage = delta + gamma * gae_lambda * continues * advantage
        advantages[t] = advantage
        next_value = values[t]
    return advantages.detach(), (advantages + values).detach()


def ppo_update(actor, critic, actor_optimizer, critic_optimizer, rollout, config):
    """Shuffle kitchens; keep each complete timestep sequence in order."""
    advantages, targets = compute_gae(
        rollout["rewards"], rollout["values"], rollout["done"], rollout["next_value"],
        config.gamma, config.gae_lambda,
    )
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    batch_size = advantages.shape[1]
    if config.minibatch_size < 1 or config.epochs < 1:
        raise ValueError("minibatch_size and epochs must be positive.")
    totals = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}
    count = 0

    for _ in range(config.epochs):
        order = torch.randperm(batch_size, device=advantages.device)
        for start in range(0, batch_size, config.minibatch_size):
            kitchens = order[start:start + config.minibatch_size]
            # Leading dimensions remain [T, selected kitchens]. The GRU is
            # recomputed with gradients through the entire sequence, from zero.
            logits, _ = actor(
                rollout["obs"][:, kitchens],
                rollout["previous_action"][:, kitchens],
                rollout["previous_reward"][:, kitchens],
                None, rollout["episode_start"][:, kitchens],
                reset_memory=rollout["reset_actor_memory"],
            )
            new_values, _ = critic(
                rollout["critic_obs"][:, kitchens],
                None, rollout["episode_start"][:, kitchens],
            )
            distribution = Categorical(logits=logits)
            new_log_prob = distribution.log_prob(rollout["actions"][:, kitchens])
            ratio = (new_log_prob - rollout["log_prob"][:, kitchens]).exp()
            selected_advantages = advantages[:, kitchens]
            ordinary = ratio * selected_advantages
            clipped = ratio.clamp(1 - config.clip, 1 + config.clip) * selected_advantages
            actor_loss = -torch.minimum(ordinary, clipped).mean()
            entropy = distribution.entropy().mean()
            critic_loss = (new_values - targets[:, kitchens]).square().mean()

            actor_optimizer.zero_grad(set_to_none=True)
            (actor_loss - config.entropy_coef * entropy).backward()
            clip_grad_norm_(actor.parameters(), config.max_grad_norm)
            actor_optimizer.step()

            critic_optimizer.zero_grad(set_to_none=True)
            (config.value_coef * critic_loss).backward()
            clip_grad_norm_(critic.parameters(), config.max_grad_norm)
            critic_optimizer.step()

            size = kitchens.numel()
            totals["actor_loss"] += actor_loss.item() * size
            totals["critic_loss"] += critic_loss.item() * size
            totals["entropy"] += entropy.item() * size
            count += size

    return {key: value / count for key, value in totals.items()}
