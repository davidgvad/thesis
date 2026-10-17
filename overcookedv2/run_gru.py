"""Train ONE GRU actor/critic across selected GPUs (or CPU workers)."""

import argparse
from dataclasses import asdict
from datetime import timedelta
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
from torch.multiprocessing import spawn

from layouts import Layout
from ppo import PPOConfig
from teammates import AssemblyLineTeammate
from train_gru import train
from v2 import OvercookedEnv


# A small example kitchen for checking the launcher, not the thesis benchmark.
# Use --layout to supply your actual kitchen and possible recipes as JSON.
EXAMPLE_LAYOUT = {
    "grid": "
    WWWWWWWWW
    O WXWOW X
    W   P   W
    W A PA  W
    WWWBWBWWW
    ",
    "recipes": [[0, 0, 0]],
}


def run_worker(worker_index, args, init_method=None):
    # A worker handles one device and its share of kitchens. All workers train
    # the same policy, worker_index is its communication ID (called its rank).
    workers = len(args.devices)
    distributed = workers > 1
    device = torch.device(args.devices[worker_index])
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    torch.set_num_threads(args.cpu_threads)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    if distributed:
        # NCCL communicates between NVIDIA GPUs; Gloo allows CPU-only testing.
        dist.init_process_group(
            backend="nccl" if device.type == "cuda" else "gloo",
            init_method=init_method, rank=worker_index, world_size=workers,
            timeout=timedelta(minutes=args.timeout_minutes),
        )
    try:
        kitchens_per_worker = args.batch_size // workers
        layout_spec = args.layout_spec
        layout = Layout.from_string([
            (layout_spec["grid"], layout_spec["recipes"])
        ] * kitchens_per_worker)
        # Move all layout tensors before creating the environment and networks.
        layout.grid = layout.grid.to(device)
        layout.agent_positions = layout.agent_positions.to(device)
        layout.recipes = layout.recipes.to(device)
        env = OvercookedEnv(
            layout, max_steps=args.max_steps, agent_view_size=args.view_radius,
            random_reset=args.random_reset,
        )
        teammate = AssemblyLineTeammate(role=args.role)

        print(f"worker={worker_index} device={device} kitchens={kitchens_per_worker}", flush=True)
        train(
            env, teammate, updates=args.updates, config=args, seed=args.seed,
            log_dir=Path(args.log_dir) / f"seed_{args.seed}", distributed=distributed,
            metadata={
                "devices": args.devices, "cpu_threads_per_worker": args.cpu_threads,
                "layout": layout_spec, "view_radius": args.view_radius,
                "teammate_role": args.role,
                "sampling_seeds": [args.seed + rank for rank in range(workers)],
            },
        )
    finally:
        if distributed:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--devices", nargs="+",
                        help="Devices sharing one policy, e.g. cuda:0 cuda:1, or cpu cpu.")
    parser.add_argument("--seed", type=int, help="Seed for this shared training run.")
    parser.add_argument("--batch-size", type=int, help="TOTAL kitchens across all devices.")
    parser.add_argument("--updates", type=int, help="Rollout/update cycles for the shared policy.")
    parser.add_argument("--minibatch-size", type=int, help="TOTAL kitchen sequences per optimizer step.")
    parser.add_argument("--max-steps", type=int, help="Steps per episode.")
    parser.add_argument("--cpu-threads", type=int, help="PyTorch CPU threads per worker.")
    parser.add_argument("--role", choices=("runner", "deliverer", "flexible"), help="Fixed teammate role.")
    parser.add_argument("--layout", help="JSON file containing grid and recipes; None uses the example.")
    parser.add_argument("--view-radius", type=int, help="Actor view radius; None gives the full map.")
    parser.add_argument("--log-dir", help="Directory for this run's logs and checkpoint.")
    parser.add_argument("--timeout-minutes", type=int, help="Distributed communication timeout.")
    parser.add_argument("--learning-rate", type=float, help="Adam learning rate.")
    parser.add_argument("--gamma", type=float, help="Future reward discount.")
    parser.add_argument("--gae-lambda", type=float, help="GAE feedback decay.")
    parser.add_argument("--clip", type=float, help="PPO probability-ratio clipping range.")
    parser.add_argument("--entropy-coef", type=float, help="Actor exploration bonus weight.")
    parser.add_argument("--value-coef", type=float, help="Critic loss weight.")
    parser.add_argument("--max-grad-norm", type=float, help="Gradient norm clipping limit.")
    parser.add_argument("--epochs", type=int, help="PPO passes over each rollout.")
    parser.add_argument("--print-every", type=int, help="Print kitchen 0 every N rollout steps; 0 disables it.")
    parser.add_argument("--random-reset", action=argparse.BooleanOptionalAction,
                        help="Randomize agent positions/directions, inventories, and pot/counter contents on reset.")
    # Defaults live only in PPOConfig. CLI values replace the selected fields.
    parser.set_defaults(**asdict(PPOConfig()))
    args = PPOConfig(**vars(parser.parse_args()))

    for name in ("batch_size", "updates", "minibatch_size", "max_steps", "cpu_threads", "epochs", "timeout_minutes"):
        if getattr(args, name) < 1:
            parser.error(f"{name.replace('_', '-')} must be positive.")
    if args.view_radius is not None and args.view_radius < 0:
        parser.error("view-radius must be nonnegative.")
    if args.print_every < 0:
        parser.error("print-every must be nonnegative.")
    workers = len(args.devices)
    # Equal splits give every worker the same number of backward calls and the
    # same weight in the averaged gradients, including the final minibatch.
    for name in ("batch_size", "minibatch_size"):
        if getattr(args, name) % workers != 0:
            parser.error(f"{name.replace('_', '-')} must divide evenly across {workers} workers.")
    device_types = set()
    gpu_indices = set()
    for name in args.devices:
        try:
            device = torch.device(name)
        except (RuntimeError, ValueError) as error:
            parser.error(str(error))
        if device.type not in ("cpu", "cuda"):
            parser.error("Choose cpu or cuda:<GPU index>.")
        device_types.add(device.type)
        if device.type == "cuda":
            if device.index is None:
                index = 0
            else:
                index = device.index
            if index >= torch.cuda.device_count():
                parser.error(f"{name} is unavailable; check your GPU allocation and PyTorch installation.")
            if index in gpu_indices:
                parser.error("Choose a different GPU for each worker.")
            gpu_indices.add(index)
    if len(device_types) != 1:
        parser.error("Use all CUDA devices or all CPU workers for one shared run.")
    if workers > 1:
        if not dist.is_available():
            parser.error("This PyTorch installation does not support distributed training.")
        if "cuda" in device_types and not dist.is_nccl_available():
            parser.error("Multi-GPU training requires PyTorch with the NCCL backend.")
        if "cpu" in device_types and not dist.is_gloo_available():
            parser.error("Multi-CPU testing requires PyTorch with the Gloo backend.")

    args.layout_spec = EXAMPLE_LAYOUT
    if args.layout is not None:
        args.layout_spec = json.loads(Path(args.layout).read_text())
    # Validate the task before starting the workers.
    sample_layout = Layout.from_string([(args.layout_spec["grid"], args.layout_spec["recipes"])])
    if sample_layout.agent_positions.shape[1] != 2:
        parser.error("The training kitchen must contain exactly two agents.")
    run_dir = Path(args.log_dir) / f"seed_{args.seed}"
    if run_dir.exists():
        parser.error(f"{run_dir} already exists; choose a new --log-dir or --seed.")
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"One shared policy, seed={args.seed}, output={run_dir}", flush=True)

    if workers == 1:
        run_worker(0, args)
    else:
        # The temporary file lets workers on THIS server find each other.
        # spawn reports worker errors and stops the other workers if one fails.
        with TemporaryDirectory(prefix="overcooked_ddp_") as directory:
            init_method = (Path(directory) / "workers").as_uri()
            spawn(run_worker, args=(args, init_method), nprocs=workers, join=True)


if __name__ == "__main__":
    main()
