# Overcooked V2: GRU actor and PPO

A batched PyTorch environment and a small recurrent PPO implementation.
The teammate design is deliberately undecided. You supply its policy as a
function; no heuristic or neural teammate is defined here.

- overcookedv2/models.py: the CNN encoder, GRU actor, and separate GRU critic.
- overcookedv2/ppo.py: critic inputs, rollout collection, GAE, and PPO updates.
- overcookedv2/train_gru.py: training, evaluation, logging, and model saving.
- overcookedv2/test_gru.py: checks for memory, gradients, inputs, and training.

The actor sees its own legal observation, previous action, previous shared
reward, and its own memory. The critic separately sees the full grid, the true
recipe, pot and button timers, both agents, and normalized remaining time.
Research labels are logging metadata only; neither network receives them.

Both networks use the CNN from the brief. Their GRUs have input and hidden
size 128. GRUCell applies one GRU layer one timestep at a time. Each kitchen
has independent actor and critic memories shaped [B, 128].

Install PyTorch and run the checks from the repository root:

    python3 -m pip install -r requirements.txt
    python3 -m unittest discover -s overcookedv2 -p 'test_gru.py' -v

The networks work with one timestep [B, H, W, C] or an ordered sequence
[T, B, H, W, C]. The actor returns six logits and its next hidden state.
The critic returns one value per kitchen and its separate next hidden state.
Use a categorical distribution to sample from the actor's logits.

From Python running in the overcookedv2 folder, use your existing environment:

    from train_gru import train, evaluate
    from ppo import PPOConfig

    # Supply teammate_policy after deciding how teammates should work.
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

The tests use a STAY-action stub solely to exercise the pipeline. It is not
an experimental teammate choice. Low/high coverage conditions, training and
held-out teammate groups, and the earlier-behavior/later-decision pilot remain
to be defined once the task and teammate designs are chosen.
