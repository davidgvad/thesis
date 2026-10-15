from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from models import GRUActor, CentralCritic
from ppo import PPOConfig, collect_rollout, ppo_update


def train(
    env, teammate, updates=10, config=None, seed=0,
    log_dir=None, save_traces=False, metadata=None,
):
    """Train the ego. The caller supplies a teammate; none is defined here.

    teammate(env, state, agent_index) must return actions shaped [B].
    Metadata is only logged, never passed to either network.
    """
    if updates < 1:
        raise ValueError("updates must be positive.")
    config = PPOConfig() if config is None else config
    torch.manual_seed(seed)
    device = env.layout.grid.device
    actor = GRUActor(env.obs_shape).to(device)
    critic_shape = (
        env.layout.grid.shape[1], env.layout.grid.shape[2], env.obs_channels + 1
    )
    critic = CentralCritic(critic_shape).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.learning_rate)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / "config.json").open("w") as file:
            json.dump({
                "ppo": asdict(config), "seed": seed, "metadata": metadata,
                "actor_obs_shape": env.obs_shape, "critic_obs_shape": critic_shape,
                "max_steps": env.max_steps,
            }, file, indent=2)

    history = []
    for update in range(1, updates + 1):
        start = time.perf_counter()
        rollout = collect_rollout(env, actor, critic, teammate)
        metrics = ppo_update(
            actor, critic, actor_optimizer, critic_optimizer, rollout, config
        )
        metrics.update({
            "update": update,
            "env_steps": update * env.layout.grid.shape[0] * env.max_steps,
            "mean_return": rollout["rewards"].sum(dim=0).mean().item(),
            "episode_returns": rollout["rewards"].sum(dim=0).cpu().tolist(),
            "seconds": time.perf_counter() - start, "seed": seed,
        })
        history.append(metrics)
        print(json.dumps(metrics), flush=True)

        if log_dir is not None:
            with (log_dir / "metrics.jsonl").open("a") as file:
                file.write(json.dumps(metrics) + "\n")
            if save_traces:
                trace_keys = (
                    "joint_actions", "rewards", "done", "episode_start",
                    "actor_hidden", "values", "action_probs",
                )
                torch.save(
                    {key: rollout[key].cpu() for key in trace_keys},
                    log_dir / f"trace_{update}.pt",
                )

    if log_dir is not None:
        torch.save({
            "actor": actor.state_dict(), "critic": critic.state_dict(),
            "actor_obs_shape": env.obs_shape, "critic_obs_shape": critic_shape,
            "ppo": asdict(config), "seed": seed, "max_steps": env.max_steps,
        }, log_dir / "checkpoint.pt")

    return actor, critic, history


@torch.no_grad()
def evaluate(env, actor, critic, teammate, episodes=1, reset_actor_memory=False):
    """Freeze parameter updates but keep within-episode memory updates."""
    if episodes < 1:
        raise ValueError("episodes must be positive.")
    actor.eval()
    critic.eval()
    returns = []
    for _ in range(episodes):
        rollout = collect_rollout(
            env, actor, critic, teammate, reset_actor_memory=reset_actor_memory
        )
        returns.append(rollout["rewards"].sum(dim=0))
    return torch.stack(returns)
