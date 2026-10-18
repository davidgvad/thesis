# Overcooked V2: GRU actor and PPO

A batched PyTorch environment and a small recurrent PPO implementation.
You can use the configurable Assembly Line heuristic or train two separate
neural actors sharing one centralized critic.

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

The heuristic interface is teammate_policy(env, state, agent_index), returning
integer actions shaped [B] on the environment's device. Neural teammates use
the actor interface described below, so their observations, previous actions,
and memories stay separate from the ego's.

## Two learning neural actors

Use `--teammate-type neural` to train a second GRU actor with its own weights
and Adam optimizer. Both actors use the same team advantages from one shared
centralized critic. Every minibatch updates actor 0, actor 1, then the critic
once. This is a MAPPO-style setup with separate actor parameters.

```bash
NCCL_P2P_DISABLE=1 python3 -u overcookedv2/run_gru.py \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 --teammate-type neural \
    --log-dir runs/two_learning_actors
```

`--teammate-type heuristic` keeps the original fixed-teammate mode and is the
default. `--role` selects the heuristic role only. No network definitions or
environment rules change between these modes.

For another compatible neural architecture, pass your own instances from Python:

```python
ego = GRUActor(env.obs_shape)        # Replace with your compatible actor class.
partner = GRUActor(env.obs_shape)    # A separate instance with separate weights.
actor, critic, history = train(
    env, actor=ego, teammate_actor=partner,
    config=PPOConfig(), log_dir="runs/custom_pair",
)
```

Passing `teammate_actor` selects neural mode. The supplied models are updated
in place, so `partner` remains available afterward. The return tuple is unchanged.
A compatible actor follows:

```text
forward(obs, previous_action, previous_reward, hidden, episode_start,
        reset_memory=False) -> (six action logits, next_hidden)
```

It accepts a single timestep [B,...] and an ordered training sequence [T,B,...].
The actor resets its own memory using episode_start. Memory is either None
(for a feed-forward policy) or a tensor with a consistent shape for stacking
the trace; it is not shared between actors. Other architectures need an adapter
if their interface differs. collect_rollout and evaluate also accept a
teammate_actor for neural-pair rollouts without any weight updates.

Neural-mode checkpoints add `teammate_actor` alongside `actor` and `critic`.
Metrics add `teammate_actor_loss` and `teammate_entropy`. The existing periodic
checks still evaluate the ego (actor 0) against the three heuristic roles;
they do not update either neural actor.

For a timing comparison, run each mode with the same settings and add
`--updates 12 --print-every 0 --eval-every 0`. Compare `rollout_seconds`,
`ppo_seconds`, and total `seconds` in metrics.jsonl after the first two updates.
CUDA work is synchronized at timing boundaries; distributed times report the
slowest worker in each phase. Training two neural actors adds both partner
inference and a second actor update, so this measures joint-training cost.

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
Every evaluation check also saves the latest weights to checkpoint.pt, replacing
the previous checkpoint. This stores model weights, not optimizer/resume state.
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

### Periodic evaluation and saved results

By default, training evaluates the frozen policy every 20 updates and at the
final update against each heuristic role: runner, deliverer, and flexible.
Evaluation uses empty pots/counters/inventories, random positions/directions
within each agent's room, the same recipes and other rules, and sampled actor
actions. Its returns can differ from training returns, which use random kitchen
contents by default. Evaluation does not update weights or consume the training
random-number sequence; the model's training/evaluation mode is restored afterward.

All workers participate using the original models (no DDP backward calls).
Their returns are gathered before worker 0 logs the global scores. Defaults live
in PPOConfig and can be overridden with:

```text
--eval-every 20    # 0 disables evaluation
--eval-batches 1   # per role: eval_batches * global batch_size episodes
--eval-seed 10000  # fixed evaluation seeds, plus worker rank
```

With the default 256 kitchens, one check evaluates 256 episodes per role,
768 episodes total. Use --eval-batches 2 for 512 episodes per role. Every
checkpoint uses the same evaluation seeds to make progress easier to compare.
Evaluation interactions are additional work and are not included in env_steps,
which counts only training interactions.

Inside the run's seed folder, training writes:

- `evaluations.jsonl`: one record per role/check, with update, training steps,
  mean/std return, episode count, individual returns, seeds, and reset mode.
- `checkpoint.pt`: the latest evaluated weights, also saved at the final update.

Use `evaluations.jsonl` for your own analysis and plotting. Training only
records and prints evaluation scores; it does not generate graphs.

These are validation checks against the existing role family, including the
training teammate's role; they are not a separate held-out thesis test set.
Episode standard deviations describe variability within one training seed.

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
The current default is 4 PPO epochs per rollout; `--updates` separately controls
how many collect-rollout/train cycles to run. The resolved settings are saved
with the run, so CLI overrides are recorded too.

`overcookedv2/run_gru.py` distributes ONE training run over devices on one server.
Each GPU has its own kitchens and copies of the training networks (two actors
and one critic in neural mode). Each network's copies start with identical
weights across workers; actor 0 and actor 1 remain distinct networks.
PyTorch DistributedDataParallel (DDP)
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

The terminal prints compact update summaries: progress, environment steps,
mean return, actor/critic losses, entropy, seed, and elapsed update time. The
individual kitchen returns are saved in `metrics.jsonl`, without printing
the long list to the terminal.

By default, `PPOConfig.print_every = 100` shows kitchen 0 on worker 0 initially,
every 100 rollout steps, and at the end of each rollout. Use `--print-every 10`
for more frequent snapshots, `--print-every 1` for every step, or `--print-every 0`
to disable them. Each bordered snapshot includes a grid with row/column labels,
agent facing arrows, a table of latest actions and inventories, the recipe,
pot/counter contents and timers, last team reward, and that kitchen's running
return. `E` is the ego and `T` is the teammate; `*` marks an item on a counter.
Positions show the state after the displayed actions. At episode end, the grid,
recipe, and inventories instead show the automatic reset, explicitly labelled;
the actions and return still belong to the episode that just ended. This is a
full-state debug view, even when the actor has a local view. Displaying GPU
tensors in the terminal adds overhead; use `--print-every 0` for timed runs.

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

The default training budget is 293 updates: `293 * 256 * 400 = 30,003,200`
environment steps per run. This matches the approximate 30M-step scale used for
several coordination tasks in the [OvercookedV2 paper, Appendix D.2](https://arxiv.org/html/2503.17821v1#A4.SS2).
The paper also uses 4 PPO epochs, 10 independent training seeds, and 500
evaluation episodes per policy pairing. Matching the step budget and epoch
count does not reproduce its full protocol; our batching, teammate training,
task, and reset distribution differ.

For the planned two-architecture by two-coverage thesis comparison, 10 seeds
per condition would mean 40 main runs (about 1.2B total training steps), versus
12 runs (about 360M steps) with the proposal's 3 seeds. These are independent
training repetitions, not the per-worker sampling seeds. Specialist-reference
training, tuning, and evaluation add to this budget. The population comparison,
second architecture, and held-out thesis evaluation protocol remain to be
implemented; the current fixed-teammate example is a training pilot with
periodic validation against all three heuristic roles.

The final evaluation target is 500 episodes per trained seed and held-out
teammate, using the same evaluation distribution across conditions. Keep test
teammates separate from validation/tuning, compare untrained/random ego and
specialist baselines, report results by recipe/teammate, and report uncertainty
across independent training seeds. Extra evaluation episodes do not replace
independent training seeds. For the current runner, `--updates 10` still gives
a short smoke test instead of the longer default run.

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

Training uses `PPOConfig.random_reset = True` by default. Each rollout starts
with freshly sampled agent positions/directions, inventories, and pot/counter
contents, using the environment's existing random reset. The map stays fixed,
and agents stay in their original rooms. Automatic resets at episode end use
the same mode. Use `--no-random-reset` for the original empty starting state,
or `--random-reset` to explicitly enable randomization. The run seed controls
the sampling sequence, and the chosen setting is saved in the configuration.

`grid` uses the symbols from `Layout.from_string` and must contain two agents.
`recipes` lists possible three-ingredient recipes. Use `--role runner`,
`--role deliverer`, or `--role flexible` (default) to choose the fixed teammate.
`--view-radius 2` gives the actor a local view; omitting it gives the full map.
The critic still receives its centralized input. Device choices, layout, role,
view radius, CPU threads, and sampling seeds are saved in the run configuration.
The launcher uses one heuristic configuration or one learning neural partner;
response-based teammate populations remain a separate part of the experiment.

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
