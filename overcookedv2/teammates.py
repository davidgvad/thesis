"""Batched PyTorch adaptation of ICRL4AHT's Assembly Line heuristic family.

Source: https://github.com/AHT-Hub/ICRL4AHT/blob/main/agents/overcooked_v2/assembly_line_agent.py
This keeps the runner / deliverer / flexible roles, not every upstream parameter.
Like that policy, it reads the full state and true recipe.
"""

import torch
import torch.nn.functional as F

from commons import Action, DynamicObject as D, StaticObject as S


def _neighbors(values, padding):
    """[B, H, W] -> [B, H, W, 4], in UP/DOWN/LEFT/RIGHT movement order.

    At each destination, read the cells that could move into it: below, above,
    right, left. Padding prevents paths wrapping around the edge of the map.
    """
    padded = F.pad(values, (1, 1, 1, 1), value=padding)
    return torch.stack([
        padded[:, 2:, 1:-1], padded[:, :-2, 1:-1],
        padded[:, 1:-1, 2:], padded[:, 1:-1, :-2],
    ], dim=-1)


def _paths(grid, start, occupied, max_distance=None):
    """Batched BFS. Inputs: grid [B,H,W,3], start [B,2], occupied [B,N-1,2].

    Return distance and first_action, both [B,H,W]. Unreachable cells have
    distance H*W+1 and action -1. All kitchens expand together each round.
    """
    B, H, W = grid.shape[:3]
    device = grid.device
    unreachable = H * W + 1
    start_cell = start[:, 0] * W + start[:, 1]
    occupied_cells = occupied[..., 0] * W + occupied[..., 1]
    floor = (grid[..., 0] == int(S.EMPTY)).flatten(1)
    floor.scatter_(1, occupied_cells, False)
    floor.scatter_(1, start_cell[:, None], True)
    floor = floor.reshape(B, H, W)

    distance = torch.full((B, H * W), unreachable, device=device, dtype=torch.long)
    distance.scatter_(1, start_cell[:, None], 0)
    distance = distance.reshape(B, H, W)
    first_action = torch.full_like(distance, 6)
    first_action.flatten(1).scatter_(1, start_cell[:, None], int(Action.STAY))
    frontier = first_action.clone()
    moves = torch.arange(4, device=device)
    rounds = H * W - 1 if max_distance is None else max_distance

    for depth in range(rounds):
        candidates = _neighbors(frontier, 6)
        candidates = torch.where(candidates == int(Action.STAY), moves, candidates)
        next_action = candidates.amin(dim=-1)
        reached = floor & (distance == unreachable) & (next_action < 4)
        distance = torch.where(reached, depth + 1, distance)
        first_action = torch.where(reached, next_action, first_action)
        frontier = torch.where(reached, next_action, 6)
        # A CPU-only early exit saves work on small maps. Never synchronize a
        # GPU here; GPU calls run the fixed, cached upper bound instead.
        if device.type == "cpu" and not reached.any():
            break

    return distance, first_action.masked_fill(first_action == 6, -1)


def _nearest(targets, costs, actions):
    """Choose an interaction target in every kitchen; -1 means unreachable."""
    B = targets.shape[0]
    scores = costs.masked_fill(~targets[..., None], 1000000)
    cost, index = scores.reshape(B, -1).min(dim=-1)
    batch = torch.arange(B, device=actions.device)
    flat_actions = actions.reshape(B, -1)
    action = flat_actions[batch, index]
    return action.masked_fill(cost >= targets.shape[1] * targets.shape[2] + 1, -1)


class AssemblyLineTeammate:
    def __init__(self, role="flexible", hesitation_prob=0.0):
        if role not in ("runner", "deliverer", "flexible"):
            raise ValueError("role must be 'runner', 'deliverer', or 'flexible'.")
        if not 0 <= hesitation_prob <= 1:
            raise ValueError("hesitation_prob must be between 0 and 1.")
        self.role = role
        self.hesitation_prob = hesitation_prob
        self._cache_key = None

    def _prepare(self, env, state):
        """Cache static layout information once, rather than every timestep."""
        B, H, W = state.grid.shape[:3]
        device = state.grid.device
        key = (id(env.layout.grid), B, H, W, device)
        if self._cache_key == key:
            return
        self._layout_grid = env.layout.grid  # Keep its identity alive with the cache.
        static = state.grid[..., 0]
        self._floor = static == int(S.EMPTY)
        self._pots = static == int(S.POT)
        self._counters = static == int(S.WALL)
        self._plates = static == int(S.PLATE_PILE)
        self._goals = static == int(S.GOAL)
        self._piles = static >= int(S.INGREDIENT_PILE_BASE)
        self._pile_ids = (static - int(S.INGREDIENT_PILE_BASE)).clamp(0, 9)
        self._batch = torch.arange(B, device=device)
        self._rows = torch.arange(H, device=device)[:, None].expand(H, W)
        self._cols = torch.arange(W, device=device)[None, :].expand(H, W)
        self._delta = env.delta[:4]
        self._open_neighbors = _neighbors(self._floor, False).sum(dim=-1)

        stations = self._pots | self._plates | self._goals | self._piles
        cells = torch.stack([self._rows, self._cols], dim=-1).reshape(-1, 2)
        manhattan = (cells[:, None] - cells[None, :]).abs().sum(dim=-1)
        station_distances = manhattan[None].masked_fill(~stations.flatten(1)[:, None], H * W + 1)
        self._station_distance = station_distances.amin(dim=-1).reshape(B, H, W)
        self._station_distance = torch.where(stations.any(dim=(1, 2))[:, None, None], self._station_distance, 0)

        # These two scalar reads happen only when preparing a new layout.
        self._max_distance = max(0, int(self._floor.sum(dim=(1, 2)).max().item()) - 1)
        self._num_pots = int(self._pots.sum(dim=(1, 2)).max().item())
        self._pot_cells = self._pots.flatten(1).long().argsort(dim=1, descending=True, stable=True)[:, :self._num_pots]
        self._cache_key = key

    @torch.no_grad()
    def __call__(self, env, state, agent_index):
        """Choose actions [B] on the state's device, without per-kitchen loops."""
        B, H, W = state.grid.shape[:3]
        N = state.agents.pos.shape[1]
        if not 0 <= agent_index < N:
            raise ValueError("agent_index is outside the available agents.")
        self._prepare(env, state)
        device = state.grid.device
        batch = self._batch
        positions = state.agents.pos
        position = positions[:, agent_index]
        direction = state.agents.direction[:, agent_index]
        inventory = state.agents.inventory
        held = inventory[:, agent_index]
        others = torch.cat([positions[:, :agent_index], positions[:, agent_index + 1:]], dim=1)
        other_items = torch.cat([inventory[:, :agent_index], inventory[:, agent_index + 1:]], dim=1)
        items, timers = state.grid[..., 1], state.grid[..., 2]
        unreachable = H * W + 1

        distance, first_action = _paths(state.grid, position, others, self._max_distance)
        neighbor_distance = _neighbors(distance, unreachable)
        neighbor_action = _neighbors(first_action, -1)
        facing = torch.arange(4, device=device)
        facing_matches = direction[:, None, None, None] == facing
        costs = neighbor_distance + ((neighbor_distance == 0) & ~facing_matches).long()
        actions = torch.where(neighbor_distance == 0,
                              torch.where(facing_matches, int(Action.INTERACT), facing), neighbor_action)

        def go(targets):
            return _nearest(targets, costs, actions)

        def counts(encoded):
            return (encoded[..., None] // env.ingredient_values) % 4

        contents = counts(items)                      # [B,H,W,10]
        required = counts(state.recipe)               # [B,10]
        inventory_counts = counts(inventory)          # [B,N,10]
        raw_inventory = (inventory != 0) & ((inventory & 3) == 0)
        carried = (inventory_counts * raw_inventory[..., None]).sum(dim=1)
        reserved = (inventory_counts[:, :agent_index] * raw_inventory[:, :agent_index, None]).sum(dim=1)
        cooked = self._pots & ((items & int(D.COOKED)) != 0)
        hot = self._pots & ((timers > 0) | cooked)
        ready = self._pots & ~hot & (items == state.recipe[:, None, None])
        filling = self._pots & ~hot & ~ready & (contents <= required[:, None, None]).all(dim=-1)

        # Counters reachable by a partner's room are preferred for handoffs.
        partner_floor = torch.zeros_like(self._floor)
        if N > 1:
            rooms = env.rooms
            partner_rooms = rooms[batch[:, None], others[..., 0], others[..., 1]]
            partner_floor = (rooms[..., None] == partner_rooms[:, None, None]).any(dim=-1) & (rooms >= 0)
        shared_counters = _neighbors(partner_floor, False).any(dim=-1)
        empty_counter = self._counters & (items == 0)
        drop = go(empty_counter & shared_counters)
        drop = torch.where(drop >= 0, drop, go(empty_counter))

        dish_action = go(self._goals)
        dish_action = torch.where(dish_action >= 0, dish_action, drop)
        plate_action = go(cooked)
        wait = go(hot)
        wait = torch.where(wait == int(Action.INTERACT), int(Action.STAY), wait)
        plate_action = torch.where(plate_action >= 0, plate_action, wait)
        plate_action = torch.where(plate_action >= 0, plate_action, drop)
        useful_pots = filling & (
            contents + inventory_counts[:, agent_index, None, None] + reserved[:, None, None]
            <= required[:, None, None]
        ).all(dim=-1)
        ingredient_action = go(useful_pots) if self.role != "deliverer" else torch.full_like(held, -1)
        ingredient_action = torch.where(ingredient_action >= 0, ingredient_action, drop)

        # Empty hands: service first, then start cooking, then prepare ingredients.
        empty_action = torch.full_like(held, -1)
        if self.role != "runner":
            staged_dishes = self._counters & ((items & 3) == 3)
            empty_action = go(staged_dishes & (go(self._goals) >= 0)[:, None, None])
        empty_action = torch.where(empty_action >= 0, empty_action, go(ready))
        if self.role != "runner":
            plate_source = self._plates | (self._counters & (items == int(D.PLATE)))
            pickup_plate = go(plate_source & hot.any(dim=(1, 2))[:, None, None])
            empty_action = torch.where(empty_action >= 0, empty_action, pickup_plate)

        if self.role != "deliverer":
            flat_contents = contents.flatten(1, 2)
            flat_filling = filling.flatten(1)
            pot_counts = flat_contents[batch[:, None], self._pot_cells]
            pot_valid = flat_filling[batch[:, None], self._pot_cells]
            priority = pot_counts.sum(dim=-1).masked_fill(~pot_valid, -1)
            order = priority.argsort(dim=1, descending=True, stable=True)
            pot_counts = pot_counts[batch[:, None], order]
            pot_valid = pot_valid[batch[:, None], order]
            partner_busy = (other_items >= int(D.BASE_INGREDIENT)).any(dim=1)
            # A small loop over pots; every iteration still covers all B kitchens.
            for pot in range(self._num_pots):
                missing = required - pot_counts[:, pot] - carried
                pile_needed = missing[batch[:, None, None], self._pile_ids] > 0
                counter_needed = self._counters & (items != 0) & ((items & 3) == 0)
                counter_needed &= (contents <= missing.clamp_min(0)[:, None, None]).all(dim=-1)
                source_action = go((self._piles & pile_needed) | counter_needed)
                prepare = (empty_action < 0) & pot_valid[:, pot]
                if self.role == "flexible":
                    prepare &= ~partner_busy
                empty_action = torch.where(prepare, source_action, empty_action)

        held_is_dish = ((held & int(D.PLATE)) != 0) & ((held & int(D.COOKED)) != 0)
        action = torch.where(held == 0, empty_action,
                             torch.where(held_is_dish, dish_action,
                                         torch.where(held == int(D.PLATE), plate_action, ingredient_action)))

        # When idle, park away from stations. Tie-break using the first move.
        parking_score = ((self._station_distance * 5 + self._open_neighbors) * unreachable - distance) * 5 - first_action
        parking_score = parking_score.masked_fill(distance == unreachable, torch.iinfo(torch.long).min)
        parking = parking_score.flatten(1).argmax(dim=1)
        park_action = first_action.flatten(1)[batch, parking]
        action = torch.where(action >= 0, action, park_action)

        if agent_index > 0:
            next_position = position + env.delta[action]
            near_earlier = ((next_position[:, None] - positions[:, :agent_index]).abs().sum(dim=-1) <= 1).any(dim=1)
            yield_turn = (action < int(Action.STAY)) & (state.time % 2 == 1) & near_earlier
            action = action.masked_fill(yield_turn, int(Action.STAY))

        # Empty agents give nearby ingredient/dish carriers room to pass.
        if N > 1:
            carrier_distances = (
                (self._rows[None, None] - others[..., 0, None, None]).abs()
                + (self._cols[None, None] - others[..., 1, None, None]).abs()
            ).masked_fill(other_items[..., None, None] < int(D.BASE_INGREDIENT), unreachable)
            carrier_distance = carrier_distances.amin(dim=1)
            current_distance = carrier_distance[batch, position[:, 0], position[:, 1]]
            escape_cells = position[:, None] + self._delta[None]
            r, c = escape_cells[..., 0].clamp(0, H - 1), escape_cells[..., 1].clamp(0, W - 1)
            escape_distance = carrier_distance[batch[:, None], r, c]
            escape_score = escape_distance * unreachable + self._station_distance[batch[:, None], r, c]
            escape_score = escape_score.masked_fill(distance[batch[:, None], r, c] != 1, -1)
            escape = escape_score.argmax(dim=1)
            can_escape = (escape_score.max(dim=1).values >= 0)
            can_escape &= escape_distance[batch, escape] > current_distance
            yield_carrier = (held == 0) & (current_distance <= 2) & can_escape
            action = torch.where(yield_carrier, escape, action)

        if self.hesitation_prob > 0:
            pause = torch.rand(B, device=device) < self.hesitation_prob
            action = action.masked_fill(pause, int(Action.STAY))
        return action
