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
