from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_

from commons import Action, DynamicObject as D, StaticObject as S


@dataclass
class PPOConfig:
    """All launcher and PPO defaults; CLI arguments can override each field."""

    # PPO settings.
    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.25
    epochs: int = 4
    minibatch_size: int = 4  # Total kitchen sequences per step, across all workers.

    # Run settings. Batch sizes are totals across all workers.
    devices: list[str] = field(default_factory=lambda: ["cpu"])
    seed: int = 0
    batch_size: int = 256
    updates: int = 293  # 30,003,200 steps with batch_size=256, max_steps=400.
    max_steps: int = 400
    cpu_threads: int = 4
    role: str = "runner"
    teammate_type: str = "heuristic"  # "neural" trains a separate second actor.
    layout: Optional[str] = None  # None selects the launcher's example kitchen.
    view_radius: Optional[int] = 3 # None gives the actor the full map.
    log_dir: str = "runs/gru"
    timeout_minutes: int = 10
    print_every: int = 100  # Print kitchen 0 every N rollout steps; 0 disables it.
    random_reset: bool = True  # Sample positions, inventories, and kitchen contents on reset.
    eval_every: int = 20  # Evaluate all three teammate roles; 0 disables evaluation.
    eval_batches: int = 1  # Full batches per role: 1 means batch_size episodes in total.
    eval_seed: int = 10000  # Reuse evaluation seeds at every check.


def print_kitchen(
    state, ego_index=0, actions=None, reward=None, done=None,
    *, step=0, steps=None, episode_return=0.0,
):
    """Print one full-state kitchen snapshot, including the last joint action."""
    grid = state.grid[0].cpu().tolist()
    positions = state.agents.pos[0].cpu().tolist()
    directions = state.agents.direction[0].cpu().tolist()
    inventory = state.agents.inventory[0].cpu().tolist()

    def item_name(item):
        if item == 0:
            return "empty"
        if item == int(D.PLATE):
            return "plate"
        kind = "dish" if item & int(D.PLATE) else "ingredients"
        return f"{kind} {D.get_ingredients_list(item)}" + (" (cooked)" if item & int(D.COOKED) else "")

    symbols = {S.EMPTY: ".", S.WALL: "#", S.POT: "P", S.PLATE_PILE: "B",
               S.GOAL: "X", S.RECIPE_INDICATOR: "R", S.BUTTON_RECIPE_INDICATOR: "L"}
    symbols.update({int(S.INGREDIENT_PILE_BASE) + i: str(i) for i in range(10)})
    board = [[symbols.get(cell[0], "?").center(3) for cell in row] for row in grid]
    for row, cells in enumerate(grid):
        for col, (static, item, _) in enumerate(cells):
            if static == int(S.WALL):
                board[row][col] = " * " if item else "###"
    for agent, (row, col) in enumerate(positions):
        label = "E" if agent == ego_index else "T"
        board[row][col] = f"{label}{'^v<>'[directions[agent]]}".center(3)

    progress = f" | Rollout step {step}/{steps}" if steps is not None else ""
    lines = [f"Kitchen 0{progress} | Episode time {state.time[0].item()}"]
    names = ["-"] * len(positions)
    if actions is not None:
        names = [Action(action).name.replace("MOVE_", "") for action in actions[0].cpu().tolist()]
    if reward is not None:
        lines.append(f"Last team reward: {reward[0].item():+g} | Kitchen return so far: {float(episode_return):g}")
    if done is not None and done[0].item():
        lines.append("EPISODE ENDED: the grid, recipe, and inventories below show the automatic reset.")
        lines.append("The displayed last actions, reward, and return belong to the episode that ended.")

    lines.extend(["", f"Recipe: {D.get_ingredients_list(state.recipe[0].item())} (full-state debug view)"])
    width = len(grid[0])
    border = "    +" + "---+" * width
    lines.append("     " + " ".join(f"{col:^3}" for col in range(width)))
    lines.append(border)
    for row, cells in enumerate(board):
        lines.append(f"{row:>3} |" + "|".join(cells) + "|")
        lines.append(border)
    lines.extend([
        "E/T = ego/teammate; ^ v < > = facing; # = counter; * = counter with item",
        "P = pot; B = plates; X = delivery; R/L = recipe; digits = ingredient piles",
        "",
        f"{'Agent':<10} {'Last action':<12} {'Position':<12} {'Facing':<8} Holding",
    ])
    for agent, position in enumerate(positions):
        label = "Ego" if agent == ego_index else "Teammate"
        facing = Action(directions[agent]).name.replace("MOVE_", "")
        lines.append(f"{label:<10} {names[agent]:<12} {str(tuple(position)):<12} {facing:<8} {item_name(inventory[agent])}")
    lines.extend(["", "Pots and items on counters (row, column):"])
    for row, cells in enumerate(grid):
        for col, (static, item, timer) in enumerate(cells):
            if item or timer or static == int(S.POT):
                lines.append(f"  ({row}, {col}) {symbols.get(static, '?')}: {item_name(item)}, timer={timer}")
    panel_width = max(len(line) for line in lines)
    panel_border = "+" + "-" * (panel_width + 2) + "+"
    panel = [panel_border] + [f"| {line.ljust(panel_width)} |" for line in lines] + [panel_border]
    print("\n" + "\n".join(panel), flush=True)


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
    env, actor, critic, teammate=None, ego_index=0, steps=None, reset_actor_memory=False,
    print_every=0, teammate_actor=None,
):
    """Collect ordered episodes. Defaults to the entire finite episode horizon.

    Supply either a fixed teammate(env, state, index), or teammate_actor using
    the same observation/history/memory interface as actor. The neural actors
    have separate memories and share one team-value prediction from critic.
    Each collection starts fresh, with zero action/reward history and memory.
    A shorter collection bootstraps its unfinished boundary; it is a truncated
    training sequence. Use the default full horizon for the pilot.
    """
    if env.layout.agent_positions.shape[1] != 2 or ego_index not in (0, 1):
        raise ValueError("This collector requires two agents and ego_index 0 or 1.")
    if teammate_actor is None and teammate is None:
        raise ValueError("Provide a heuristic teammate or a teammate_actor.")
    steps = env.max_steps if steps is None else steps
    if steps < 1 or steps > env.max_steps:
        raise ValueError("steps must be between 1 and the episode horizon.")
    if print_every < 0:
        raise ValueError("print_every must be nonnegative.")

    state = env.reset()
    if print_every:
        print_kitchen(state, ego_index, step=0, steps=steps)
    obs = env.get_obs(state)
    batch_size = state.grid.shape[0]
    device = state.grid.device
    kitchen_return = torch.zeros((), device=device) if print_every else None
    previous_action = torch.zeros(batch_size, dtype=torch.long, device=device)
    previous_reward = torch.zeros(batch_size, device=device)
    episode_start = torch.ones(batch_size, dtype=torch.bool, device=device)
    actor_hidden = None
    teammate_hidden = None
    previous_teammate_action = torch.zeros_like(previous_action)
    critic_hidden = None
    data = {key: [] for key in (
        "obs", "critic_obs", "previous_action", "previous_reward", "episode_start",
        "actions", "joint_actions", "log_prob", "values", "rewards", "done",
        "actor_hidden", "action_probs",
    )}
    # Only the second actor's own inputs/outputs differ. Rewards, boundaries,
    # and centralized values are shared by both actors.
    teammate_data = {key: [] for key in (
        "obs", "previous_action", "actions", "log_prob", "actor_hidden", "action_probs",
    )} if teammate_actor is not None else None

    for step in range(steps):
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
        if teammate_actor is None:
            joint_actions[:, 1 - ego_index] = teammate(env, state, 1 - ego_index)
        else:
            teammate_obs = obs[:, 1 - ego_index]
            teammate_logits, teammate_hidden = teammate_actor(
                teammate_obs, previous_teammate_action, previous_reward,
                teammate_hidden, episode_start,
            )
            teammate_distribution = Categorical(logits=teammate_logits)
            teammate_action = teammate_distribution.sample()
            joint_actions[:, 1 - ego_index] = teammate_action
            teammate_transition = {
                "obs": teammate_obs, "previous_action": previous_teammate_action,
                "actions": teammate_action,
                "log_prob": teammate_distribution.log_prob(teammate_action),
                "actor_hidden": teammate_hidden, "action_probs": teammate_distribution.probs,
            }
            for key, value in teammate_transition.items():
                teammate_data[key].append(value)
        next_state, next_obs, reward, _, done = env.step(state, joint_actions)
        # Shaped rewards are deliberately excluded from the training objective.
        if print_every:
            kitchen_return += reward[0]
            if (step + 1) % print_every == 0 or step + 1 == steps:
                print_kitchen(
                    next_state, ego_index, joint_actions, reward, done,
                    step=step + 1, steps=steps, episode_return=kitchen_return.item(),
                )
            kitchen_return = kitchen_return.masked_fill(done[0], 0)

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
        previous_teammate_action = joint_actions[:, 1 - ego_index].masked_fill(done, 0)
        previous_reward = reward.masked_fill(done, 0)
        # Each network resets its own memory using episode_start on the next call.

    next_value, _ = critic(
        get_critic_obs(env, state, ego_index), critic_hidden, episode_start
    )
    rollout = {
        key: torch.stack(values) if values[0] is not None else None
        for key, values in data.items()
    }
    # A reset kitchen's value belongs to a different episode: never bootstrap it.
    rollout["next_value"] = next_value.masked_fill(episode_start, 0)
    rollout["reset_actor_memory"] = reset_actor_memory
    if teammate_data is not None:
        # Copy the shared fields, then replace the actor-specific fields.
        teammate_rollout = rollout.copy()
        for key, values in teammate_data.items():
            teammate_rollout[key] = torch.stack(values) if values[0] is not None else None
        teammate_rollout["reset_actor_memory"] = False
        rollout["teammate"] = teammate_rollout
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


def normalize_advantages(advantages, distributed=False):
    """Use one mean/std for the entire rollout, including other workers."""
    if not distributed:
        return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    # all_reduce adds together each worker's numbers and returns the sum to all.
    totals = torch.stack([advantages.sum(), advantages.new_tensor(advantages.numel())])
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    mean = totals[0] / totals[1]
    squared_error = (advantages - mean).square().sum()
    dist.all_reduce(squared_error, op=dist.ReduceOp.SUM)
    std = (squared_error / totals[1]).sqrt()
    return (advantages - mean) / (std + 1e-8)


def _update_actor(actor, optimizer, rollout, kitchens, advantages, config):
    """One PPO actor step; reusable for either actor's own trajectory."""
    logits, _ = actor(
        rollout["obs"][:, kitchens],
        rollout["previous_action"][:, kitchens],
        rollout["previous_reward"][:, kitchens],
        None, rollout["episode_start"][:, kitchens],
        reset_memory=rollout["reset_actor_memory"],
    )
    distribution = Categorical(logits=logits)
    new_log_prob = distribution.log_prob(rollout["actions"][:, kitchens])
    ratio = (new_log_prob - rollout["log_prob"][:, kitchens]).exp()
    selected_advantages = advantages[:, kitchens]
    ordinary = ratio * selected_advantages
    clipped = ratio.clamp(1 - config.clip, 1 + config.clip) * selected_advantages
    actor_loss = -torch.minimum(ordinary, clipped).mean()
    entropy = distribution.entropy().mean()

    optimizer.zero_grad(set_to_none=True)
    # DDP synchronizes this actor's gradients before clipping and stepping.
    (actor_loss - config.entropy_coef * entropy).backward()
    clip_grad_norm_(actor.parameters(), config.max_grad_norm)
    optimizer.step()
    return actor_loss.item(), entropy.item()


def ppo_update(
    actor, critic, actor_optimizer, critic_optimizer, rollout, config,
    distributed=False, teammate_actor=None, teammate_optimizer=None,
):
    """Keep sequences ordered; DDP averages gradients during backward().

    In distributed mode, every worker must have the same rollout shape/config.
    Train a second actor when its model and optimizer are supplied. Both actors
    use the same team advantages; the critic is updated only once per minibatch.
    All networks being updated must be DDP wrappers in distributed mode.
    """
    minibatch_size = config.minibatch_size
    if minibatch_size < 1 or config.epochs < 1:
        raise ValueError("minibatch_size and epochs must be positive.")
    if (teammate_actor is None) != (teammate_optimizer is None):
        raise ValueError("Provide both teammate_actor and teammate_optimizer to train it.")
    if teammate_actor is not None and "teammate" not in rollout:
        raise ValueError("The rollout must contain the neural teammate's experience.")
    if distributed:
        if minibatch_size % dist.get_world_size() != 0:
            raise ValueError("minibatch_size must divide evenly across workers.")
        # Each worker handles only its share of the total minibatch.
        minibatch_size //= dist.get_world_size()
    advantages, targets = compute_gae(
        rollout["rewards"], rollout["values"], rollout["done"], rollout["next_value"],
        config.gamma, config.gae_lambda,
    )
    advantages = normalize_advantages(advantages, distributed)
    batch_size = advantages.shape[1]
    totals = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}
    if teammate_actor is not None:
        totals.update({"teammate_actor_loss": 0.0, "teammate_entropy": 0.0})
    count = 0

    for _ in range(config.epochs):
        order = torch.randperm(batch_size, device=advantages.device)
        for start in range(0, batch_size, minibatch_size):
            kitchens = order[start:start + minibatch_size]
            # Leading dimensions remain [T, selected kitchens]. The GRU is
            # recomputed with gradients through the entire sequence, from zero.
            actor_loss, entropy = _update_actor(
                actor, actor_optimizer, rollout, kitchens, advantages, config,
            )
            size = kitchens.numel()
            if teammate_actor is not None:
                teammate_loss, teammate_entropy = _update_actor(
                    teammate_actor, teammate_optimizer, rollout["teammate"],
                    kitchens, advantages, config,
                )
                totals["teammate_actor_loss"] += teammate_loss * size
                totals["teammate_entropy"] += teammate_entropy * size

            new_values, _ = critic(
                rollout["critic_obs"][:, kitchens],
                None, rollout["episode_start"][:, kitchens],
            )
            critic_loss = (new_values - targets[:, kitchens]).square().mean()

            critic_optimizer.zero_grad(set_to_none=True)
            (config.value_coef * critic_loss).backward()
            clip_grad_norm_(critic.parameters(), config.max_grad_norm)
            critic_optimizer.step()

            totals["actor_loss"] += actor_loss * size
            totals["critic_loss"] += critic_loss.item() * size
            totals["entropy"] += entropy * size
            count += size

    if distributed:
        # Report losses over all workers, weighted by their kitchen counts.
        combined = torch.tensor(
            list(totals.values()) + [count],
            device=advantages.device, dtype=torch.float64,
        )
        dist.all_reduce(combined, op=dist.ReduceOp.SUM)
        numbers = combined.tolist()
        totals = dict(zip(totals, numbers[:-1]))
        count = numbers[-1]
    return {key: value / count for key, value in totals.items()}
