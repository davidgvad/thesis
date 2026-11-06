"""A batched clue-and-memory task, separate from the Overcooked environment.

Observation: [clue_A_visible, clue_B_visible, decision_required].
Actions: A=0, B=1, WAIT=2. Only WAIT is legal before the final decision.
The hidden condition in MemoryState is for the simulator/researcher, not the actor.
"""

import argparse
from dataclasses import dataclass, replace

import torch


CHOOSE_A, CHOOSE_B, WAIT = 0, 1, 2
ACTION_NAMES = ("A", "B", "WAIT")


@dataclass
class MemoryState:
    condition: torch.Tensor  # [B]: hidden correct answer, A=0 or B=1.
    clue: torch.Tensor       # [B]: possibly noisy clue, sampled once per episode.
    time: int                # All episodes in this batch advance together.


class MemoryPOMDP:
    def __init__(
        self, batch_size=256, irrelevant_steps=4, delay_steps=5,
        clue_accuracy=1.0, clue_timing="early", device="cpu", seed=0,
    ):
        if batch_size < 1 or irrelevant_steps < 0 or delay_steps < 0:
            raise ValueError("Use a positive batch_size and nonnegative phase lengths.")
        if not 0.5 <= clue_accuracy <= 1.0:
            raise ValueError("clue_accuracy must be between 0.5 and 1.0.")
        if clue_timing not in ("early", "late"):
            raise ValueError("clue_timing must be 'early' or 'late'.")
        self.batch_size = batch_size
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.clue_accuracy = clue_accuracy
        self.clue_timing = clue_timing
        self.decision_step = irrelevant_steps + 1 + delay_steps
        self.max_steps = self.decision_step + 1
        # Late clues appear in the terminal observation, after the choice is locked.
        self.clue_step = irrelevant_steps if clue_timing == "early" else self.max_steps
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    def reset(self):
        """Sample independent A/B conditions and start a fresh batch at time zero."""
        condition = torch.randint(
            2, (self.batch_size,), device=self.device, generator=self.generator,
        )
        correct_clue = torch.rand(
            self.batch_size, device=self.device, generator=self.generator,
        ) < self.clue_accuracy
        clue = torch.where(correct_clue, condition, 1 - condition)
        return MemoryState(condition=condition, clue=clue, time=0)

    def get_obs(self, state):
        """Return only legal observations [B,3]; never expose the hidden answer."""
        obs = torch.zeros((self.batch_size, 3), device=self.device)
        if state.time == self.clue_step:
            batch = torch.arange(self.batch_size, device=self.device)
            obs[batch, state.clue] = 1
        if state.time == self.decision_step:
            obs[:, 2] = 1
        return obs

    def action_mask(self, state):
        """True means legal. A learned policy should mask other logits to -inf."""
        allowed = torch.zeros((self.batch_size, 3), dtype=torch.bool, device=self.device)
        if state.time == self.decision_step:
            allowed[:, :2] = True
        elif state.time < self.max_steps:
            allowed[:, WAIT] = True
        return allowed

    def step(self, state, actions):
        """Return (next_state, next_obs, reward [B], done [B]).

        There is no automatic reset: inspect terminal results, then call reset().
        WAIT is forced before the choice, preventing previous-action memory tricks.
        """
        if state.time >= self.max_steps:
            raise ValueError("The episode has finished; call reset() before stepping again.")
        if actions.shape != (self.batch_size,) or actions.dtype != torch.long:
            raise ValueError("actions must be a torch.long tensor shaped [batch_size].")
        if actions.device != self.device:
            raise ValueError("actions must be on the environment's device.")
        if state.time == self.decision_step:
            legal = (actions == CHOOSE_A) | (actions == CHOOSE_B)
        else:
            legal = actions == WAIT
        if not legal.all().item():
            raise ValueError("Only WAIT is legal before the decision; choose A or B at the decision.")

        reward = torch.zeros(self.batch_size, device=self.device)
        if state.time == self.decision_step:
            reward = (actions == state.condition).float()
        next_state = replace(state, time=state.time + 1)
        done = torch.full(
            (self.batch_size,), next_state.time == self.max_steps,
            dtype=torch.bool, device=self.device,
        )
        return next_state, self.get_obs(next_state), reward, done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--delay-steps", type=int, default=5)
    parser.add_argument("--clue-accuracy", type=float, default=1.0)
    parser.add_argument("--clue-timing", choices=("early", "late"), default="early")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    env = MemoryPOMDP(**vars(parser.parse_args()))
    state = env.reset()
    remembered = torch.zeros(env.batch_size, dtype=torch.long, device=env.device)
    print("Episode 0 trace: observations are [clue A, clue B, choose now].")
    print("This is a scripted sanity check, not a trained neural policy.")

    for _ in range(env.max_steps):
        obs = env.get_obs(state)
        clue_visible = obs[:, :2].any(dim=-1)
        remembered = torch.where(clue_visible, obs[:, :2].argmax(dim=-1), remembered)
        actions = torch.full_like(remembered, WAIT)
        if state.time == env.decision_step:
            actions = remembered
            # Same decision states, but a policy that ignores history always chooses A.
            _, _, memoryless_reward, _ = env.step(state, torch.zeros_like(actions))
        next_state, next_obs, reward, done = env.step(state, actions)
        print(
            f"t={state.time:2d} obs={obs[0].tolist()} "
            f"action={ACTION_NAMES[actions[0].item()]:4s} reward={reward[0].item():.0f}"
        )
        state = next_state

    print(f"Final observation: {next_obs[0].tolist()}; done={done[0].item()}")
    print(f"Mean return, remembering the clue: {reward.mean().item():.3f}")
    print(f"Mean return, always choosing A:    {memoryless_reward.mean().item():.3f}")
    optimal = env.clue_accuracy if env.clue_timing == "early" else 0.5
    print(f"Optimal expected return: {optimal:.3f}; guessing: 0.500")


if __name__ == "__main__":
    main()
