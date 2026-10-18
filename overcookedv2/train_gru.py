from dataclasses import asdict, replace
import copy
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from models import GRUActor, CentralCritic
from ppo import PPOConfig, collect_rollout, ppo_update
from teammates import AssemblyLineTeammate


def train(
    env, teammate=None, updates=None, config=None, seed=None,
    log_dir=None, save_traces=False, metadata=None,
    distributed=False, actor=None, teammate_actor=None,
):
    """Train one actor with a heuristic, or two separate actors with one critic.

    Supply custom actor/teammate_actor models using the GRUActor interface.
    A supplied teammate_actor selects neural mode and is updated in place.
    Otherwise config.teammate_type chooses a heuristic or a fresh GRU partner.
    The return stays (actor, critic, history); both actors are saved in neural mode.
    Omitted updates and seed use their values from config.
    Metadata is only logged, never passed to either network.
    distributed=True requires an initialized process group and equal local
    rollout sizes. The launcher handles that; ordinary Python calls default off.
    """
    config = PPOConfig() if config is None else config
    if teammate_actor is not None:
        config = replace(config, teammate_type="neural")
    if config.teammate_type not in ("heuristic", "neural"):
        raise ValueError("teammate_type must be 'heuristic' or 'neural'.")
    updates = config.updates if updates is None else updates
    seed = config.seed if seed is None else seed
    if updates < 1:
        raise ValueError("updates must be positive.")
    if config.eval_every < 0 or config.eval_batches < 1:
        raise ValueError("eval_every must be nonnegative and eval_batches must be positive.")
    rank = dist.get_rank() if distributed else 0
    workers = dist.get_world_size() if distributed else 1
    is_main = rank == 0
    torch.manual_seed(seed)
    device = env.layout.grid.device
    actor = GRUActor(env.obs_shape) if actor is None else actor
    actor = actor.to(device)
    critic_shape = (
        env.layout.grid.shape[1], env.layout.grid.shape[2], env.obs_channels + 1
    )
    critic = CentralCritic(critic_shape).to(device)
    if config.teammate_type == "neural":
        teammate_actor = GRUActor(env.obs_shape) if teammate_actor is None else teammate_actor
        teammate_actor = teammate_actor.to(device)
        actor_parameters = {id(parameter) for parameter in actor.parameters()}
        if actor_parameters.intersection(id(parameter) for parameter in teammate_actor.parameters()):
            raise ValueError("The two actors must have separate parameters.")
    elif teammate is None:
        raise ValueError("Provide a heuristic teammate, or select teammate_type='neural'.")

    # Keep the original models for rollout inference, saving, and evaluation.
    # DDP copies worker 0's initial weights to every worker, then synchronizes
    # gradients whenever the wrapped models run backward() in ppo_update().
    actor_for_update, critic_for_update = actor, critic
    teammate_for_update = teammate_actor
    if distributed:
        device_ids = [device.index] if device.type == "cuda" else None
        actor_for_update = DDP(actor, device_ids=device_ids, broadcast_buffers=False)
        critic_for_update = DDP(critic, device_ids=device_ids, broadcast_buffers=False)
        if teammate_actor is not None:
            teammate_for_update = DDP(teammate_actor, device_ids=device_ids, broadcast_buffers=False)
        # One experiment seed, but different action/recipe/shuffle draws per worker.
        torch.manual_seed(seed + rank)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=config.learning_rate)
    teammate_optimizer = None
    if teammate_actor is not None:
        teammate_optimizer = torch.optim.Adam(teammate_actor.parameters(), lr=config.learning_rate)

    # Same layouts and rules, but evaluate cooking from empty kitchens.
    # The environment keeps state in the rollout, so only these settings differ.
    eval_env = copy.copy(env)
    eval_env.random_reset = False
    eval_env.random_agent_positions = True

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
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        if is_main and config.print_every:
            print(f"\nRollout for update {update}", flush=True)
        rollout = collect_rollout(
            env, actor, critic, teammate,
            print_every=config.print_every if is_main else 0,
            teammate_actor=teammate_actor,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        rollout_seconds = time.perf_counter() - start
        ppo_start = time.perf_counter()
        metrics = ppo_update(
            actor_for_update, critic_for_update, actor_optimizer, critic_optimizer,
            rollout, config, distributed=distributed,
            teammate_actor=teammate_for_update, teammate_optimizer=teammate_optimizer,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ppo_seconds = time.perf_counter() - ppo_start
        episode_returns = rollout["rewards"].sum(dim=0)
        if distributed:
            # Gather each worker's [local B] returns to report the global batch.
            returns_by_worker = [torch.empty_like(episode_returns) for _ in range(workers)]
            dist.all_gather(returns_by_worker, episode_returns)
            episode_returns = torch.cat(returns_by_worker)
        seconds = time.perf_counter() - start
        if distributed:
            # An update is only finished when its slowest worker is finished.
            worker_seconds = torch.tensor(
                [seconds, rollout_seconds, ppo_seconds], device=device, dtype=torch.float64,
            )
            dist.all_reduce(worker_seconds, op=dist.ReduceOp.MAX)
            seconds, rollout_seconds, ppo_seconds = worker_seconds.tolist()
        metrics.update({
            "update": update,
            "env_steps": update * env.layout.grid.shape[0] * workers * env.max_steps,
            "mean_return": episode_returns.mean().item(),
            "episode_returns": episode_returns.cpu().tolist(),
            "seconds": seconds, "seed": seed,
            "rollout_seconds": rollout_seconds, "ppo_seconds": ppo_seconds,
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
            if teammate_actor is not None:
                print(
                    f"  Teammate actor loss {metrics['teammate_actor_loss']:.5f} | "
                    f"Teammate entropy {metrics['teammate_entropy']:.3f}", flush=True,
                )
            print(f"  Rollout {rollout_seconds:.2f}s | PPO {ppo_seconds:.2f}s", flush=True)

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
                    {key: rollout[key].cpu() if rollout[key] is not None else None for key in trace_keys},
                    log_dir / trace_name,
                )

        evaluate_now = config.eval_every > 0 and (
            update % config.eval_every == 0 or update == updates
        )
        if evaluate_now:
            if is_main:
                print(f"\nEvaluation at update {update} (empty kitchens, random positions)", flush=True)
            for role in ("runner", "deliverer", "flexible"):
                returns = evaluate(
                    eval_env, actor, critic, AssemblyLineTeammate(role),
                    episodes=config.eval_batches, seed=config.eval_seed + rank,
                ).flatten()
                if distributed:
                    # Each GPU evaluates its own kitchens, then shares the returns.
                    returns_by_worker = [torch.empty_like(returns) for _ in range(workers)]
                    dist.all_gather(returns_by_worker, returns)
                    returns = torch.cat(returns_by_worker)
                if is_main:
                    result = {
                        "update": update, "env_steps": metrics["env_steps"],
                        "seed": seed, "eval_seed": config.eval_seed, "workers": workers,
                        "teammate_role": role, "episodes": returns.numel(),
                        "mean_return": returns.mean().item(),
                        "std_return": returns.std(unbiased=False).item(),
                        "episode_returns": returns.cpu().tolist(),
                        "reset_mode": "empty_kitchens_random_positions",
                    }
                    print(
                        f"  {role:<10} | Mean return {result['mean_return']:.2f} | "
                        f"Std {result['std_return']:.2f} | Episodes {result['episodes']}",
                        flush=True,
                    )
                    if log_dir is not None:
                        with (log_dir / "evaluations.jsonl").open("a") as file:
                            file.write(json.dumps(result) + "\n")
        if log_dir is not None and is_main and (evaluate_now or update == updates):
            # Keep the latest evaluated weights, also saved at the end of training.
            temporary_checkpoint = log_dir / "checkpoint.tmp"
            checkpoint = {
                "actor": actor.state_dict(), "critic": critic.state_dict(),
                "actor_obs_shape": env.obs_shape, "critic_obs_shape": critic_shape,
                "ppo": asdict(config), "seed": seed, "max_steps": env.max_steps,
                "workers": workers, "update": update, "env_steps": metrics["env_steps"],
            }
            if teammate_actor is not None:
                checkpoint["teammate_actor"] = teammate_actor.state_dict()
            torch.save(checkpoint, temporary_checkpoint)
            temporary_checkpoint.replace(log_dir / "checkpoint.pt")
        if distributed and evaluate_now:
            # Wait for worker 0 to finish writing the checkpoint.
            dist.barrier()

    return actor, critic, history


@torch.no_grad()
def evaluate(
    env, actor, critic, teammate=None, episodes=1, reset_actor_memory=False,
    seed=None, teammate_actor=None,
):
    """Evaluate full batches without changing weights or training randomness.

    episodes is the number of batches: result shape is [episodes, local B].
    Each rollout starts with fresh recurrent memory; actions remain sampled.
    """
    if episodes < 1:
        raise ValueError("episodes must be positive.")
    models = [actor, critic]
    if teammate_actor is not None:
        models.append(teammate_actor)
    training_modes = [model.training for model in models]
    for model in models:
        model.eval()
    device = env.layout.grid.device
    cuda_devices = [device] if device.type == "cuda" else []
    try:
        # Restore the training RNG states after evaluation, including on CUDA.
        with torch.random.fork_rng(devices=cuda_devices):
            if seed is not None:
                torch.random.default_generator.manual_seed(seed)
                if device.type == "cuda":
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(seed)
            returns = []
            for _ in range(episodes):
                rollout = collect_rollout(
                    env, actor, critic, teammate, reset_actor_memory=reset_actor_memory,
                    teammate_actor=teammate_actor,
                )
                returns.append(rollout["rewards"].sum(dim=0))
            return torch.stack(returns)
    finally:
        for model, was_training in zip(models, training_modes):
            model.train(was_training)
