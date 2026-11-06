"""Train a toy ego, then measure history information and probe its frozen memory."""

import argparse
import json
from pathlib import Path

import torch
from torch.distributions import Categorical

from toy_pomdp import MemoryPOMDP
from run_toy import train_toy, evaluate_memory_reset
from history_classifier import (
    HistoryClassifier, split_by_episode, fit_classifier, evaluate_classifier,
)
from linear_probe import LinearProbe, NonlinearProbe, fit_probe, evaluate_probe


@torch.no_grad()
def collect_diagnostics(actor, env):
    """Record one legal history and decision representation per fresh episode."""
    state = env.reset()
    previous_action = torch.zeros(env.batch_size, dtype=torch.long, device=env.device)
    previous_reward = torch.zeros(env.batch_size, device=env.device)
    starts = torch.ones(env.batch_size, dtype=torch.bool, device=env.device)
    hidden = None
    observations, actions, rewards = [], [], []

    for _ in range(env.max_steps):
        obs = env.get_obs(state)
        observations.append(obs.cpu())
        actions.append(previous_action.cpu())
        rewards.append(previous_reward.cpu())
        logits, hidden = actor(obs, previous_action, previous_reward, hidden, starts)
        if state.time == env.decision_step:
            # Save memory before sampling the final action or seeing its reward.
            representation = hidden["representation"] if isinstance(hidden, dict) else hidden
            representations = representation.detach().cpu()
            labels = state.condition.cpu().clone()  # A/B targets, never actor inputs.
        allowed = env.action_mask(state)
        distribution = Categorical(logits=logits.masked_fill(~allowed, -torch.inf))
        action = distribution.sample()
        state, _, reward, done = env.step(state, action)
        previous_action, previous_reward, starts = action, reward, done
        # The terminal observation (including a late clue) is never recorded.

    data = {
        "obs": torch.stack(observations),
        "previous_action": torch.stack(actions),
        "previous_reward": torch.stack(rewards),
        "lengths": torch.full((env.batch_size,), env.max_steps, dtype=torch.long),
        "labels": labels,
        "episode_ids": torch.arange(env.batch_size),
    }
    return data, representations, reward.cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clue-timing", choices=("early", "late"), default="early")
    parser.add_argument("--clue-accuracy", type=float, default=1.0)
    parser.add_argument("--irrelevant-steps", type=int, default=4)
    parser.add_argument("--delay-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256, help="Parallel PPO environments.")
    parser.add_argument("--num-steps", type=int, default=200, help="Steps collected per PPO update.")
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument("--actor-type", choices=("gru", "transformer"), default="gru")
    parser.add_argument("--context-length", type=int, default=400)
    parser.add_argument("--reset-before-decision", action="store_true",
                        help="Also compare the frozen ego with its decision-time memory cleared.")
    parser.add_argument("--diagnostic-episodes", type=int, default=2048)
    parser.add_argument("--diagnostic-epochs", type=int, default=10)
    parser.add_argument("--diagnostic-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs/toy_experiment"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    env_settings = {
        "clue_timing": args.clue_timing, "clue_accuracy": args.clue_accuracy,
        "irrelevant_steps": args.irrelevant_steps, "delay_steps": args.delay_steps,
        "device": args.device,
    }

    # 1. Train the selected shared actor and GRU critic using rewards only.
    env = MemoryPOMDP(batch_size=args.batch_size, seed=args.seed, **env_settings)
    actor, critic, training = train_toy(
        env, updates=args.updates, num_steps=args.num_steps,
        actor_type=args.actor_type, context_length=args.context_length,
    )

    # 2. Freeze the ego and collect new episodes, separate from PPO training.
    for model in (actor, critic):
        model.eval()
        model.requires_grad_(False)
    torch.manual_seed(args.seed + 20000)
    diagnostic_env = MemoryPOMDP(
        batch_size=args.diagnostic_episodes, seed=args.seed + 10000, **env_settings,
    )
    data, representations, returns = collect_diagnostics(actor, diagnostic_env)
    train, test = split_by_episode(data, seed=args.seed)
    # Here each episode has one example, so its ID is also its representation row.
    train_ids, test_ids = train["episode_ids"], test["episode_ids"]
    scores = {
        "ego_mean_return": returns[test_ids].mean().item(),
        "optimal_expected_return": args.clue_accuracy if args.clue_timing == "early" else 0.5,
    }
    weights = {"actor": actor.state_dict(), "critic": critic.state_dict()}
    print(f"\nFrozen ego mean return: {scores['ego_mean_return']:.3f}", flush=True)

    # 3. Can the current observation or the full legal history predict A/B?
    for name, current_only in (("current_observation", True), ("history", False)):
        torch.manual_seed(args.seed)
        classifier = HistoryClassifier(
            obs_shape=(3,), num_classes=2, num_actions=3,
            max_steps=env.max_steps, current_only=current_only,
        ).to(args.device)
        fit_classifier(
            classifier, train, epochs=args.diagnostic_epochs,
            batch_size=args.diagnostic_batch_size, seed=args.seed,
        )
        result = evaluate_classifier(classifier, test, batch_size=args.diagnostic_batch_size)
        scores[f"{name}_accuracy"] = result["accuracy"]
        weights[name] = classifier.state_dict()
        print(f"{name:20s} accuracy: {result['accuracy']:.3f}", flush=True)

    # 4. Can a linear or nonlinear decoder recover A/B from the ego's memory?
    for name, probe_class in (("linear_probe", LinearProbe), ("nonlinear_probe", NonlinearProbe)):
        torch.manual_seed(args.seed)
        probe = probe_class(num_classes=2).to(args.device)
        fit_probe(
            probe, representations[train_ids], train["labels"],
            epochs=args.diagnostic_epochs, batch_size=args.diagnostic_batch_size,
        )
        accuracy = evaluate_probe(probe, representations[test_ids], test["labels"])
        scores[f"{name}_accuracy"] = accuracy
        weights[name] = probe.state_dict()
        print(f"{name:20s} accuracy: {accuracy:.3f}", flush=True)

    majority = train["labels"].bincount(minlength=2).argmax()
    scores["majority_baseline_accuracy"] = (test["labels"] == majority).float().mean().item()
    memory_reset = None
    if args.reset_before_decision:
        diagnostic_env.generator.manual_seed(args.seed + 10000)
        torch.manual_seed(args.seed + 20000)
        memory_reset = evaluate_memory_reset(actor, diagnostic_env)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "results.json").open("w") as file:
        json.dump({"settings": settings, "scores": scores, "training": training,
                   "memory_reset": memory_reset,
                   "diagnostic_train_episodes": train_ids.numel(),
                   "diagnostic_test_episodes": test_ids.numel()}, file, indent=2)
    torch.save({
        "settings": settings, "weights": weights, "histories": data,
        "representations": representations, "episode_returns": returns,
        "memory_reset": memory_reset,
        "train_ids": train_ids, "test_ids": test_ids,
    }, args.output / "experiment.pt")
    print(f"Saved results and models to {args.output}")


if __name__ == "__main__":
    main()
