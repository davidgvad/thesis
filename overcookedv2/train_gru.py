from dataclasses import asdict
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from models import GRUActor, CentralCritic
from ppo import PPOConfig, collect_rollout, ppo_update


def train(
    env, teammate, updates=None, config=None, seed=None,
    log_dir=None, save_traces=False, metadata=None,
    distributed=False,
):
    """Train the ego. The caller supplies a teammate; none is defined here.

    teammate(env, state, agent_index) must return actions shaped [B].
    Omitted updates and seed use their values from config.
    Metadata is only logged, never passed to either network.
    distributed=True requires an initialized process group and equal local
    rollout sizes. The launcher handles that; ordinary Python calls default off.
    """
    config = PPOConfig() if config is None else config
    updates = config.updates if updates is None else updates
    seed = config.seed if seed is None else seed
    if updates < 1:
        raise ValueError("updates must be positive.")
    rank = dist.get_rank() if distributed else 0
    workers = dist.get_world_size() if distributed else 1
    is_main = rank == 0
    torch.manual_seed(seed)
    device = env.layout.grid.device
    actor = GRUActor(env.obs_shape).to(device)
    critic_shape = (
        env.layout.grid.shape[1], env.layout.grid.shape[2], env.obs_channels + 1
    )
    critic = CentralCritic(critic_shape).to(device)

    # Keep the original models for rollout inference, saving, and evaluation.
    # DDP copies worker 0's initial weights to every worker, then synchronizes
    # gradients whenever the wrapped models run backward() in ppo_update().
    actor_for_update, critic_for_update = actor, critic
    if distributed:
        device_ids = [device.index] if device.type == "cuda" else None
        actor_for_update = DDP(actor, device_ids=device_ids, broadcast_buffers=False)
        critic_for_update = DDP(critic, device_ids=device_ids, broadcast_buffers=False)
        # One experiment seed, but different action/recipe/shuffle draws per worker.
        torch.manual_seed(seed + rank)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.learning_rate)

    if log_dir is not None:
        log_dir = Path(log_dir)
        if is_main:
            log_dir.mkdir(parents=True, exist_ok=True)
            with (log_dir / "config.json").open("w") as file:
                json.dump({
                    "ppo": asdict(config), "seed": seed, "metadata": metadata,
                    "actor_obs_shape": env.obs_shape, "critic_obs_shape": critic_shape,
                    "max_steps": env.max_steps, "workers": workers,
                    "batch_size": env.layout.grid.shape[0] * workers,
                }, file, indent=2)
        if distributed:
            # Other workers may save traces, so wait until the folder exists.
            dist.barrier()

    history = []
    for update in range(1, updates + 1):
        start = time.perf_counter()
        if is_main and config.print_every:
            print(f"\nRollout for update {update}", flush=True)
        rollout = collect_rollout(
            env, actor, critic, teammate,
            print_every=config.print_every if is_main else 0,
        )
        metrics = ppo_update(
            actor_for_update, critic_for_update, actor_optimizer, critic_optimizer,
            rollout, config, distributed=distributed,
        )
        episode_returns = rollout["rewards"].sum(dim=0)
        if distributed:
            # Gather each worker's [local B] returns to report the global batch.
            returns_by_worker = [torch.empty_like(episode_returns) for _ in range(workers)]
            dist.all_gather(returns_by_worker, episode_returns)
            episode_returns = torch.cat(returns_by_worker)
        seconds = time.perf_counter() - start
        if distributed:
            # An update is only finished when its slowest worker is finished.
            worker_seconds = torch.tensor(seconds, device=device, dtype=torch.float64)
            dist.all_reduce(worker_seconds, op=dist.ReduceOp.MAX)
            seconds = worker_seconds.item()
        metrics.update({
            "update": update,
            "env_steps": update * env.layout.grid.shape[0] * workers * env.max_steps,
            "mean_return": episode_returns.mean().item(),
            "episode_returns": episode_returns.cpu().tolist(),
            "seconds": seconds, "seed": seed,
        })
        history.append(metrics)
        if is_main:
            print(
                f"\nUpdate {update}/{updates} | Steps {metrics['env_steps']:,} | "
                f"Mean return {metrics['mean_return']:.2f} | {seconds:.2f}s\n"
                f"  Actor loss {metrics['actor_loss']:.5f} | "
                f"Critic loss {metrics['critic_loss']:.4f} | "
                f"Entropy {metrics['entropy']:.3f} | Seed {seed}",
                flush=True,
            )

        if log_dir is not None:
            if is_main:
                with (log_dir / "metrics.jsonl").open("a") as file:
                    file.write(json.dumps(metrics) + "\n")
            if save_traces:
                trace_keys = (
                    "joint_actions", "rewards", "done", "episode_start",
                    "actor_hidden", "values", "action_probs",
                )
                trace_name = f"trace_{update}_rank_{rank}.pt" if distributed else f"trace_{update}.pt"
                torch.save(
                    {key: rollout[key].cpu() for key in trace_keys},
                    log_dir / trace_name,
                )

    if log_dir is not None and is_main:
        # Save the original models, so loading needs no DDP wrapper or prefix.
        torch.save({
            "actor": actor.state_dict(), "critic": critic.state_dict(),
            "actor_obs_shape": env.obs_shape, "critic_obs_shape": critic_shape,
            "ppo": asdict(config), "seed": seed, "max_steps": env.max_steps,
            "workers": workers,
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
