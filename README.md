# Overcooked V2: GRU actor and PPO

A batched PyTorch environment and a small recurrent PPO implementation.
You can supply a teammate policy or use the configurable Assembly Line heuristic.

- overcookedv2/models.py: the CNN encoder, GRU actor, and separate GRU critic.
- overcookedv2/ppo.py: critic inputs, rollout collection, GAE, and PPO updates.
- overcookedv2/train_gru.py: training, evaluation, logging, and model saving.
- overcookedv2/teammates.py: the Assembly Line heuristic teammate.
- overcookedv2/test_teammates.py: deliveries, routing, batching, and PPO integration checks.

The actor sees its own legal observation, previous action, previous shared
reward, and its own memory. The critic separately sees the full grid, the true
recipe, pot and button timers, both agents, and normalized remaining time.
Research labels are logging metadata only; neither network receives them.

Both networks use the CNN from the brief. Their GRUs have input and hidden
size 128. GRUCell applies one GRU layer one timestep at a time. Each kitchen
has independent actor and critic memories shaped [B, 128].

Install PyTorch and run the checks from the repository root:

    python3 -m pip install -r requirements.txt
    python3 -m unittest discover -s overcookedv2 -p 'test_*.py' -v

The networks work with one timestep [B, H, W, C] or an ordered sequence
[T, B, H, W, C]. The actor returns six logits and its next hidden state.
The critic returns one value per kitchen and its separate next hidden state.
Use a categorical distribution to sample from the actor's logits.

From Python running in the overcookedv2 folder, use your existing environment:

    from train_gru import train, evaluate
    from ppo import PPOConfig
    from teammates import AssemblyLineTeammate

    teammate_policy = AssemblyLineTeammate(role="runner", hesitation_prob=0.05)
    actor, critic, history = train(
        env, teammate_policy, updates=10, config=PPOConfig(),
        log_dir="runs/gru", save_traces=True,
    )
    returns = evaluate(env, actor, critic, teammate_policy, episodes=1)

The policy interface is teammate_policy(env, state, agent_index). It returns
integer actions shaped [B], on the environment's device. A neural teammate
can be supplied through this interface. This implementation updates only the
ego actor and centralized critic; jointly training a teammate would need its
own update step later. A recurrent teammate should reset its own memory for
kitchens where state.time == 0.

The heuristic adapts the existing
[ICRL4AHT Assembly Line family](https://github.com/AHT-Hub/ICRL4AHT/blob/main/agents/overcooked_v2/assembly_line_agent.py),
described in [the paper's Appendix B.2.4](https://arxiv.org/html/2605.24423v1).
This is a simplified implementation for this environment, not an exact reproduction
of every upstream parameter or its benchmark results.

Choose a role when creating the teammate:

| role | Main work |
| --- | --- |
| `"runner"` | Fetch recipe ingredients, fill pots, and start cooking. |
| `"deliverer"` | Fetch plates, collect cooked dishes, and deliver them. |
| `"flexible"` | Prepare ingredients and handle plating/delivery as needed. |

`hesitation_prob` is the probability of choosing STAY on each step. Its default
is zero. These are fixed rules, with no trained weights or recurrent memory.
The heuristic handles ingredient IDs 0–9, three-ingredient recipes, counter
pickups/drops, explicit cooking interactions, and basic yielding to partners.
It uses BFS to walk around walls and currently occupied cells. Roles still need
access to their stations; this does not guarantee solving every layout or
recovering every incompatible pot created by a random reset.

Like the source Assembly Line policy, this teammate reads the full state and
true recipe. It does not need to press a recipe button. The ego's actor still
receives only its legal observation. Pathfinding and action selection use batched
PyTorch operations on the state's device, without converting kitchens to CPU lists.
Static station maps and layout information are cached on the first call. That
preparation reads two scalar sizes; later GPU calls do not read tensor values
back into Python.

`_paths` now receives `grid [B,H,W,3]`, `start [B,2]`, and other agents' positions
`occupied [B,N-1,2]`. It returns `distance [B,H,W]` and `first_action [B,H,W]`.
Unreachable cells have distance `H*W+1` and action `-1`. BFS still uses successive
expansion rounds, but each round covers all kitchens and cells at once. A small
loop over pots also covers the whole batch. There is no Python loop over kitchens
or cells during action selection. For small CPU batches the extra tensor operations
can cost more than a Python search; benchmark the batch size and device you use.

For a quick environment check, two heuristic policies can control both agents:

    import torch

    runner = AssemblyLineTeammate("runner")
    deliverer = AssemblyLineTeammate("deliverer")
    state = env.reset()
    actions = torch.stack([runner(env, state, 0), deliverer(env, state, 1)], dim=-1)
    state, obs, reward, shaped_rewards, done = env.step(state, actions)

For the pilot, use two agents, 16 parallel kitchens, and max_steps=400.
Each update collects whole episodes and shuffles kitchen sequences while
keeping timesteps in order. The CNN encodes timesteps together; the GRU
processes them in order with gradients through memory. Episode-start flags
reset memory and zero previous action/reward features. The resulting done
flags describe transitions and are stored separately. Value bootstrap is zero
at episode completion. Shaped rewards are excluded from the objective.

PPO defaults: learning rate 0.00025, discount 0.99, GAE lambda 0.95, clipping
0.2, entropy coefficient 0.01, value coefficient 0.5, maximum gradient norm
0.25, and four update epochs. The debugging minibatch is four complete
kitchen sequences. Measure throughput and memory before increasing it.

For more direct control, use collect_rollout and ppo_update from ppo.py.
Each collection starts from a fresh reset and zero ego/critic memory.
collect_rollout supports steps < env.max_steps for debugging and bootstraps
the unfinished boundary; this truncates training history. Use full episodes
for the pilot.

With log_dir set, train writes metrics.jsonl, config.json, and checkpoint.pt.
save_traces=True also saves actions, rewards, episode boundaries, actor
decision representations, critic predictions, and action probabilities as
PyTorch tensors for representation probes. Optional metadata is saved only
in config.json.

Evaluation performs no parameter updates while allowing within-episode memory
updates. For an actor-memory ablation:

    returns = evaluate(
        env, actor, critic, teammate_policy, reset_actor_memory=True,
    )

This resets actor memory at every timestep while preserving its current
observation and previous-action/reward inputs.

The tests also pair complementary heuristics and run a GRU/PPO update with a real
heuristic teammate. Low/high coverage conditions and held-out teammate groups
remain to be defined and validated. Changing a role or hesitation alone does not
establish that two teammates require different complementary ego responses.

## Training one shared policy on several GPUs

All launcher and PPO defaults now live in `PPOConfig` in `overcookedv2/ppo.py`.
The launcher reads those defaults and lets command-line arguments override them.
For example, `--learning-rate 0.0001 --epochs 4 --batch-size 128` changes only
those three settings. Omitted arguments keep their `PPOConfig` defaults. Run
`python3 overcookedv2/run_gru.py --help` to list every option and its default.
The current default is 10 PPO epochs per rollout; `--updates` separately controls
how many collect-rollout/train cycles to run. The resolved settings are saved
with the run, so CLI overrides are recorded too.

`overcookedv2/run_gru.py` distributes ONE training run over devices on one server.
Each GPU has its own kitchens and a copy of the actor and critic. Those network
copies start with identical weights. PyTorch DistributedDataParallel (DDP)
averages their gradients during each `backward()` call. Every copy then applies
that same update, so the weights stay synchronized.

From the repository root, run this command **once** inside your GPU allocation:

```bash
python3 overcookedv2/run_gru.py --devices cuda:0 cuda:1 cuda:2 cuda:3 \
    --batch-size 128 --minibatch-size 16 --updates 10 \
    --cpu-threads 4 --log-dir runs/shared_gpu_pilot
```

The batch settings are **totals across all devices**. This changes the earlier
independent-seed launcher's meaning:

| Setting | Whole run | Each of four GPUs |
| --- | --- | --- |
| Kitchens per rollout (`--batch-size`) | 128 | 32 |
| Kitchen sequences per optimizer step (`--minibatch-size`) | 16 | 4 |
| Timesteps per kitchen (`--max-steps`) | 400 | 400 |

Both batch sizes must be positive and divisible by the number of workers. Each
worker runs the same number of PPO epochs and optimizer steps. A smaller final
minibatch is allowed and is split equally too. The rollout batch keeps all
kitchens' timesteps ordered; each worker shuffles its own kitchen sequences.
Advantages use the mean and standard deviation across the **whole** rollout,
not a separate normalization per GPU.

There is one experiment seed (`--seed 0` by default), with worker sampling seeds
`seed + worker_index` so workers collect different trajectories. These worker
seeds are NOT independent experimental repetitions: all data trains the same
policy. To repeat the experiment independently, launch another complete run with
`--seed 1`, then `--seed 2`, etc.

Only worker 0 writes the combined `metrics.jsonl`, `config.json`, and
`checkpoint.pt` in `runs/shared_gpu_pilot/seed_0/`. Metrics include rewards and
losses from all workers. The checkpoint stores ordinary actor/critic weights and
loads into the existing models without a DDP wrapper. If using `train(...,
save_traces=True, distributed=True)` directly, each worker saves its local traces
with a separate `_rank_0`, `_rank_1`, etc. suffix.

The step count is `updates * batch_size * max_steps` across all GPUs. With a
128-kitchen total batch and 400 steps, `--updates 586` collects approximately 30M
environment steps for the shared policy. Start with the shorter pilot to measure
speed and memory first. Increasing the GPU count does not automatically improve
throughput; the small networks and communication overhead need benchmarking.

GPU indices refer to the GPUs visible to your job. The launcher does not reserve
GPUs or CPU cores. `--cpu-threads` sets PyTorch CPU threads **per worker**; it is
not a CPU-core reservation. CUDA workers use NCCL for communication. CPU-only
workers use Gloo to test the same synchronization code. Mixing CPU and GPU workers
in one shared training run is intentionally unsupported here. If a worker fails,
the launcher reports the error and stops the others. Use a new output directory
or seed for another attempt; existing seed folders are refused to protect logs.

The default kitchen is a small example, not the selected thesis task. Supply your
actual kitchen with `--layout path/to/kitchen.json`. The JSON structure is:

```json
{"grid": "WWWWWWW\nW0 P BW\nWA R AW\nW  X  W\nWWWWWWW", "recipes": [[0, 0, 0]]}
```

`grid` uses the symbols from `Layout.from_string` and must contain two agents.
`recipes` lists possible three-ingredient recipes. Use `--role runner`,
`--role deliverer`, or `--role flexible` (default) to choose the fixed teammate.
`--view-radius 2` gives the actor a local view; omitting it gives the full map.
The critic still receives its centralized input. Device choices, layout, role,
view radius, CPU threads, and sampling seeds are saved in the run configuration.
This launcher still uses one fixed teammate configuration; response-based
teammate populations remain a separate part of the experiment.

A small CPU test of the shared-policy launcher:

```bash
python3 overcookedv2/run_gru.py --devices cpu cpu --batch-size 4 \
    --minibatch-size 2 --max-steps 4 --updates 2 --cpu-threads 1 \
    --log-dir runs/shared_cpu_check
```

Single-device training also works: pass `--devices cpu` or `--devices cuda:0`.
Existing calls to `train(env, teammate, ...)` continue to use one device by
default. The launcher initializes distributed communication and passes
`distributed=True` only when multiple devices are selected.

For reading the changes, start with `run_gru.py` (split kitchens and connect
workers), then `train_gru.py` (create DDP wrappers and combine logging), then
`ppo.py` (global advantage normalization and the usual PPO updates). During
rollout collection each worker uses its local model directly; gradients are
synchronized only during training updates.
