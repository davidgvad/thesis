"""Train a GRU or Transformer ego with a separate GRU critic on the toy POMDP."""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical

from models import GRUActor, TransformerActor, CentralCritic
from ppo import compute_gae
from toy_pomdp import MemoryPOMDP, WAIT


def train_toy(env, updates=30, num_steps=200, actor_type="gru", context_length=400):
    learning_rate = 3e-4
    epochs = 4
    minibatch_size = 64  # Complete environment sequences per optimizer step.
    clip = 0.2
    device = env.device
    batch_size = env.batch_size
    if actor_type == "transformer":
        actor = TransformerActor((3,), context_length=context_length, num_actions=3)
    else:
        actor = GRUActor((3,), num_actions=3)
    actor = actor.to(device)
    critic = CentralCritic(obs_shape=(3,)).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate)

    state = env.reset()
    previous_action = torch.zeros(batch_size, dtype=torch.long, device=device)
    previous_reward = torch.zeros(batch_size, device=device)
    episode_start = torch.ones(batch_size, dtype=torch.bool, device=device)
    actor_hidden = critic_hidden = None
    history = []
    print(f"Actor={actor_type} | Critic=gru", flush=True)

    for update in range(1, updates + 1):
        if isinstance(actor_hidden, dict):
            # PPO changed the encoder: refresh cached features from the raw history.
            actor_hidden = {key: value for key, value in actor_hidden.items() if key != "tokens"}
        initial_actor_hidden = actor_hidden
        initial_critic_hidden = critic_hidden
        data = {key: [] for key in (
            "obs", "previous_action", "previous_reward", "episode_start",
            "actions", "action_mask", "log_prob", "values", "rewards", "done",
        )}
        completed_returns = []

        # 1. Play num_steps without updating any weights.
        with torch.no_grad():
            for _ in range(num_steps):
                obs = env.get_obs(state)
                allowed = env.action_mask(state)
                logits, actor_hidden = actor(
                    obs, previous_action, previous_reward, actor_hidden, episode_start,
                )
                values, critic_hidden = critic(obs, critic_hidden, episode_start)
                distribution = Categorical(logits=logits.masked_fill(~allowed, -torch.inf))
                actions = distribution.sample()
                next_state, _, rewards, done = env.step(state, actions)
                transition = {
                    "obs": obs, "previous_action": previous_action,
                    "previous_reward": previous_reward, "episode_start": episode_start,
                    "actions": actions, "action_mask": allowed,
                    "log_prob": distribution.log_prob(actions), "values": values,
                    "rewards": rewards, "done": done,
                }
                for key, value in transition.items():
                    data[key].append(value)

                state = next_state
                if state.time == env.max_steps:
                    # The toy's only reward is the final 0/1 outcome.
                    completed_returns.append(rewards)
                    state = env.reset()  # Never feed the late terminal clue to the actor.
                episode_start = done
                previous_action = actions.masked_fill(done, 0)
                previous_reward = rewards.masked_fill(done, 0)
                # GRUs reset at episode_start; Transformer attention blocks older episodes.

            next_value, _ = critic(env.get_obs(state), critic_hidden, episode_start)
            next_value = next_value.masked_fill(episode_start, 0)

        rollout = {key: torch.stack(values) for key, values in data.items()}
        advantages, targets = compute_gae(
            rollout["rewards"], rollout["values"], rollout["done"], next_value,
        )
        decisions = rollout["action_mask"][..., 0]  # A is legal only at a real choice.
        has_decisions = decisions.any().item()
        # Rewards are already 0/1, so keep advantages on their original scale.

        # 2. Reuse the same recorded rollout for several PPO epochs.
        for _ in range(epochs):
            order = torch.randperm(batch_size, device=device)
            for indices in order.split(minibatch_size):
                starts = rollout["episode_start"][:, indices]
                if has_decisions:
                    hidden = initial_actor_hidden
                    if isinstance(hidden, dict):
                        hidden = {key: value[indices] for key, value in hidden.items()}
                    elif hidden is not None:
                        hidden = hidden[indices]
                    logits, _ = actor(
                        rollout["obs"][:, indices], rollout["previous_action"][:, indices],
                        rollout["previous_reward"][:, indices], hidden, starts,
                    )
                    # Use exactly the same legal actions as during collection.
                    logits = logits.masked_fill(~rollout["action_mask"][:, indices], -torch.inf)
                    distribution = Categorical(logits=logits)
                    new_log_prob = distribution.log_prob(rollout["actions"][:, indices])
                    ratio = (new_log_prob - rollout["log_prob"][:, indices]).exp()
                    ordinary = ratio * advantages[:, indices]
                    clipped = ratio.clamp(1 - clip, 1 + clip) * advantages[:, indices]
                    choose = decisions[:, indices]
                    # WAIT is forced; only A/B choices train the actor.
                    actor_loss = -torch.minimum(ordinary, clipped)[choose].mean()
                    entropy = distribution.entropy()[choose].mean()
                    actor_optimizer.zero_grad(set_to_none=True)
                    (actor_loss - 0.01 * entropy).backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                    actor_optimizer.step()

                hidden = initial_critic_hidden[indices] if initial_critic_hidden is not None else None
                values, _ = critic(rollout["obs"][:, indices], hidden, starts)
                critic_loss = (values - targets[:, indices]).square().mean()
                critic_optimizer.zero_grad(set_to_none=True)
                (0.5 * critic_loss).backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                critic_optimizer.step()

        mean_return = torch.stack(completed_returns).mean().item() if completed_returns else None
        history.append({
            "update": update, "env_steps": update * num_steps * batch_size,
            "mean_return": mean_return,
        })
        score = f"{mean_return:.3f}" if mean_return is not None else "n/a"
        print(f"Update {update}/{updates} | Steps {update * num_steps * batch_size:,} | Mean return {score}", flush=True)

    return actor, critic, history


@torch.no_grad()
def evaluate_memory_reset(actor, env):
    """Compare intact and cleared memory on the same fresh A/B decisions."""
    was_training = actor.training
    actor.eval()
    state = env.reset()
    previous_action = torch.zeros(env.batch_size, dtype=torch.long, device=env.device)
    previous_reward = torch.zeros(env.batch_size, device=env.device)
    starts = torch.ones(env.batch_size, dtype=torch.bool, device=env.device)
    hidden = None

    # Follow the normal policy until just before the final choice.
    for _ in range(env.decision_step):
        _, hidden = actor(env.get_obs(state), previous_action, previous_reward, hidden, starts)
        action = torch.full_like(previous_action, WAIT)  # Forced before the choice.
        state, _, previous_reward, starts = env.step(state, action)
        previous_action = action

    obs = env.get_obs(state)
    normal_logits, _ = actor(obs, previous_action, previous_reward, hidden, starts)
    reset_hidden = None  # A GRU starts with a zero hidden state.
    if isinstance(hidden, dict):
        # Transformer: remove old inputs, but keep the current timestep's position.
        reset_hidden = {
            key: value[:, :0] for key, value in hidden.items()
            if key not in ("next_position", "representation")
        }
        reset_hidden["next_position"] = hidden["next_position"]
    reset_logits, _ = actor(obs, previous_action, previous_reward, reset_hidden, starts)

    # Use the same random draw for both A/B choices to make a paired comparison.
    draw = torch.rand(env.batch_size, device=env.device)
    batch = torch.arange(env.batch_size, device=env.device)
    results = {"episodes": env.batch_size}
    print(f"\nFrozen memory comparison ({env.batch_size} episodes)", flush=True)
    for name, logits in (("intact", normal_logits), ("reset", reset_logits)):
        probabilities = logits[:, :2].softmax(dim=-1)
        action = (draw >= probabilities[:, 0]).long()
        _, _, reward, _ = env.step(state, action)
        results[name] = {
            "mean_return": reward.mean().item(),
            "mean_correct_probability": probabilities[batch, state.condition].mean().item(),
        }
        print(f"  {name:6s} | Mean return {results[name]['mean_return']:.3f}"
              f" | P(correct) {results[name]['mean_correct_probability']:.3f}", flush=True)
    actor.train(was_training)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument("--actor-type", choices=("gru", "transformer"), default="gru")
    parser.add_argument("--context-length", type=int, default=400)
    parser.add_argument("--reset-before-decision", action="store_true",
                        help="After training, compare frozen performance with and without memory.")
    parser.add_argument("--delay-steps", type=int, default=5)
    parser.add_argument("--clue-timing", choices=("early", "late"), default="early")
    parser.add_argument("--clue-accuracy", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = Path(f"runs/toy_{args.actor_type}.pt")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    env = MemoryPOMDP(
        batch_size=args.batch_size, delay_steps=args.delay_steps,
        clue_timing=args.clue_timing, clue_accuracy=args.clue_accuracy,
        device=args.device, seed=args.seed,
    )
    actor, critic, history = train_toy(
        env, updates=args.updates, num_steps=args.num_steps,
        actor_type=args.actor_type, context_length=args.context_length,
    )
    memory_reset = None
    if args.reset_before_decision:
        env.generator.manual_seed(args.seed + 10000)
        torch.manual_seed(args.seed + 20000)
        memory_reset = evaluate_memory_reset(actor, env)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "actor": actor.state_dict(), "critic": critic.state_dict(), "history": history,
        "memory_reset": memory_reset,
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }, args.output)
    print(f"Saved actor and critic to {args.output}")


if __name__ == "__main__":
    main()
