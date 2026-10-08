from dataclasses import dataclass

import torch

from commons import Action, Agents, DynamicObject as D, StaticObject as S


# Defaults from the reference Overcooked V2 environment.
DELIVERY_REWARD = 20
INDICATOR_ACTIVATION_TIME = 10
INDICATOR_ACTIVATION_COST = 5
SHAPED_REWARDS = {
    "PLACEMENT_IN_POT": 3,
    "POT_START_COOKING": 5,
    "DISH_PICKUP": 5,
    "PLATE_PICKUP": 3,
}


@dataclass
class State:
    agents: Agents
    grid: torch.Tensor        # [B, H, W, 3]: static, item, timer
    time: torch.Tensor        # [B]
    terminal: torch.Tensor    # [B], bool
    recipe: torch.Tensor      # [B]: encoded recipe
    new_correct_delivery: torch.Tensor  # [B], bool


class OvercookedEnv:
    def __init__(
        self,
        layout,
        max_steps=400,
        cook_time=20,
        negative_rewards=False,
        sample_recipe_on_delivery=False,
        random_reset=False,
        random_agent_positions=False,
    ):
        self.layout = layout
        self.max_steps = max_steps
        self.cook_time = cook_time
        self.negative_rewards = negative_rewards
        self.sample_recipe_on_delivery = sample_recipe_on_delivery
        self.random_reset = random_reset
        self.random_agent_positions = random_agent_positions

        self.delta = torch.tensor(
            [
                [-1,  0],  # UP
                [ 1,  0],  # DOWN
                [ 0, -1],  # LEFT
                [ 0,  1],  # RIGHT
                [ 0,  0],  # STAY
                [ 0,  0],  # INTERACT
            ],
            device=layout.grid.device,
        )

        # Calculate these once, not on every reset.
        self.rooms = self._find_rooms(layout.grid)
        pile_base = int(S.INGREDIENT_PILE_BASE)
        max_pile_id = (layout.grid - pile_base).clamp_min(0).max().item()
        max_recipe_id = layout.recipes.max().item()
        self.num_ingredients = int(max(max_pile_id, max_recipe_id)) + 1

        if self.num_ingredients > 10 or (layout.recipes < 0).any():
            raise ValueError("Ingredient IDs must be between 0 and 9.")

        # ASCII layouts support ten ingredient types; recipes still hold three.
        ingredient_ids = torch.arange(
            10,
            dtype=torch.long,
            device=layout.grid.device,
        )
        self.ingredient_values = 4 << (2 * ingredient_ids)

    def _encode_ingredients(self, ingredient_ids):
        values = self.ingredient_values[ingredient_ids]
        return values.sum(dim=-1)

    def reset(self, batch_indices=slice(None)):
        """Reset selected kitchens using the constructor's reset settings."""
        static = self.layout.grid[batch_indices]
        recipes = self.layout.recipes[batch_indices]
        B, H, W = static.shape
        N = self.layout.agent_positions.shape[1]
        device = static.device

        # The reference grid has three channels.
        grid = torch.stack(
            [
                static,
                torch.zeros_like(static),  # Items
                torch.zeros_like(static),  # Pot timer
            ],
            dim=-1,
        )

        batch = torch.arange(B, device=device)
        recipe_idx = torch.randint(
            recipes.shape[1],
            (B,),
            device=device,
        )
        recipe_ids = recipes[batch, recipe_idx]

        state = State(
            agents=Agents(
                pos=self.layout.agent_positions[batch_indices].clone(),
                inventory=torch.zeros((B, N), dtype=torch.long, device=device),
                direction=torch.full(
                    (B, N),
                    int(Action.MOVE_UP),
                    dtype=torch.long,
                    device=device,
                ),
            ),
            grid=grid,
            recipe=self._encode_ingredients(recipe_ids),
            time=torch.zeros(B, dtype=torch.long, device=device),
            terminal=torch.zeros(B, dtype=torch.bool, device=device),
            new_correct_delivery=torch.zeros(B, dtype=torch.bool, device=device),
        )

        if self.random_reset:
            return self._randomize_state(state, batch_indices)

        if self.random_agent_positions:
            return self._randomize_positions(state, batch_indices)

        return state

    def _find_rooms(self, static_grids):
        """Find connected floor areas, this runs once in __init__."""
        B, H, W = static_grids.shape
        rooms = torch.full((B, H, W), -1, dtype=torch.long)

        for b in range(B):
            grid = static_grids[b].cpu().tolist()
            room_number = 0

            for row in range(H):
                for col in range(W):
                    if grid[row][col] != int(S.EMPTY):
                        continue
                    if rooms[b, row, col] != -1:
                        continue

                    stack = [(row, col)]
                    rooms[b, row, col] = room_number

                    while stack:
                        r, c = stack.pop()

                        for nr, nc in (
                            (r - 1, c),
                            (r + 1, c),
                            (r, c - 1),
                            (r, c + 1),
                        ):
                            if not (0 <= nr < H and 0 <= nc < W):
                                continue
                            if grid[nr][nc] != int(S.EMPTY):
                                continue
                            if rooms[b, nr, nc] != -1:
                                continue

                            rooms[b, nr, nc] = room_number
                            stack.append((nr, nc))

                    room_number += 1

        return rooms.to(static_grids.device)

    def _randomize_positions(self, state, batch_indices=slice(None)):
        B, H, W = state.grid.shape[:3]
        device = state.grid.device
        batch = torch.arange(B, device=device)

        # Keep each agent in the room containing its layout spawn.
        spawns = self.layout.agent_positions[batch_indices]
        rooms = self.rooms[batch_indices]
        spawn_rooms = rooms[
            batch[:, None],
            spawns[..., 0],
            spawns[..., 1],
        ]

        positions = state.agents.pos.clone()
        taken = torch.zeros((B, H * W), dtype=torch.bool, device=device)
        flat_rooms = rooms.reshape(B, H * W)
        N = state.agents.pos.shape[1]

        # N shoudlnt be that big, so its not too scary to use a loop
        for agent in range(N):
            allowed = (flat_rooms == spawn_rooms[:, agent, None]) & ~taken

            scores = torch.rand((B, H * W), device=device)
            scores = scores.masked_fill(~allowed, -1)
            cell = scores.argmax(dim=-1)

            positions[:, agent, 0] = cell // W
            positions[:, agent, 1] = cell % W
            taken.scatter_(1, cell[:, None], True)

        state.agents.pos = positions
        state.agents.direction = torch.randint(0, 4, (B, N), device=device)
        return state

    def _randomize_state(self, state, batch_indices=slice(None)):
        state = self._randomize_positions(state, batch_indices)

        grid = state.grid.clone()

        B, _, _, _ = grid.shape
        N = state.agents.pos.shape[1]
        recipes = self.layout.recipes[batch_indices]
        R = recipes.shape[1]
        device = grid.device

        # ---------------------------------------------------------
        # Random agent inventories
        # ---------------------------------------------------------
        # kind 0: empty      50%
        # kind 1: plate      10%
        # kind 2: ingredient 25%
        # kind 3: dish       15%

        inventory_kind = torch.multinomial(
            torch.tensor(
                [0.50, 0.10, 0.25, 0.15],
                device=device,
            ),
            B * N,
            replacement=True,
        ).reshape(B, N)

        # Choose one random ingredient for every agent
        ingredient_ids = torch.randint(
            self.num_ingredients,
            (B, N),
            device=device,
        )

        ingredients = self.ingredient_values[ingredient_ids]

        # Choose one random recipe for every agents possible dish
        batch = torch.arange(B, device=device)[:, None]

        dish_recipe_idx = torch.randint(
            R,
            (B, N),
            device=device,
        )

        dish_ingredient_ids = recipes[
            batch,
            dish_recipe_idx,
        ]

        dishes = (
            # Does this function take in batches?
            self._encode_ingredients(dish_ingredient_ids)
            | int(D.PLATE)
            | int(D.COOKED)
        )

        state.agents.inventory = torch.where(
            inventory_kind == 1,
            int(D.PLATE),
            torch.where(
                inventory_kind == 2,
                ingredients,
                torch.where(
                    inventory_kind == 3,
                    dishes,
                    int(D.EMPTY),
                ),
            ),
        )

        # ---------------------------------------------------------
        # Random pot contents
        # ---------------------------------------------------------

        pot_b, pot_row, pot_col = (grid[..., 0] == int(S.POT)).nonzero(as_tuple=True)

        num_pots = pot_b.numel()

        if num_pots > 0:
            # kind 0: empty   40%
            # kind 1: idle    35%
            # kind 2: cooking 15%
            # kind 3: cooked  10%

            pot_kind = torch.multinomial(
                torch.tensor(
                    [0.40, 0.35, 0.15, 0.10],
                    device=device,
                ),
                num_pots,
                replacement=True,
            )

            # Generate three possible ingredients for each pot.
            pot_ingredient_ids = torch.randint(
                self.num_ingredients,
                (num_pots, 3),
                device=device,
            )

            pot_ingredients = self.ingredient_values[
                pot_ingredient_ids
            ]

            # Each nonempty pot contains between one and three ingredients.
            ingredient_count = torch.randint(
                1,
                4,
                (num_pots,),
                device=device,
            )

            ingredient_positions = torch.arange(
                3,
                device=device,
            )

            included = (
                ingredient_positions[None]
                < ingredient_count[:, None]
            )

            raw_pot_items = (
                pot_ingredients * included
            ).sum(dim=-1)

            cooked_pot_items = (
                raw_pot_items | int(D.COOKED)
            )

            grid[pot_b, pot_row, pot_col, 1] = torch.where(
                pot_kind == 0,
                int(D.EMPTY),
                torch.where(
                    pot_kind == 3,
                    cooked_pot_items,
                    raw_pot_items,
                ),
            )

            random_timer = torch.randint(
                1,
                self.cook_time + 1,
                (num_pots,),
                device=device,
            )

            grid[pot_b, pot_row, pot_col, 2] = torch.where(
                pot_kind == 2,
                random_timer,
                0,
            )

        # ---------------------------------------------------------
        # Random counter contents (which are bascially walls)
        # ---------------------------------------------------------

        wall_b, wall_row, wall_col = (
            grid[..., 0] == int(S.WALL)
        ).nonzero(as_tuple=True)

        num_walls = wall_b.numel()

        if num_walls > 0:
            # kind 0: empty      50%
            # kind 1: plate      10%
            # kind 2: ingredient 30%
            # kind 3: dish       10%

            counter_kind = torch.multinomial(
                torch.tensor(
                    [0.50, 0.10, 0.30, 0.10],
                    device=device,
                ),
                num_walls,
                replacement=True,
            )

            counter_ingredient_ids = torch.randint(
                self.num_ingredients,
                (num_walls,),
                device=device,
            )

            counter_ingredients = self.ingredient_values[
                counter_ingredient_ids
            ]

            counter_recipe_idx = torch.randint(
                R,
                (num_walls,),
                device=device,
            )

            counter_dish_ids = recipes[
                wall_b,
                counter_recipe_idx,
            ]

            counter_dishes = (
                self._encode_ingredients(counter_dish_ids)
                | int(D.PLATE)
                | int(D.COOKED)
            )

            counter_items = torch.where(
                counter_kind == 1,
                int(D.PLATE),
                torch.where(
                    counter_kind == 2,
                    counter_ingredients,
                    torch.where(
                        counter_kind == 3,
                        counter_dishes,
                        int(D.EMPTY),
                    ),
                ),
            )

            grid[wall_b, wall_row, wall_col, 1] = counter_items

        state.grid = grid

        return state

    def step(self, state, actions):
        """
        actions: [B, N]

        Returns:
            next_state
            reward:         [B]
            shaped_rewards: [B, N]
            done:           [B], True for episodes ending on this step

        Finished kitchens are reset in next_state using the configured mode.
        done still reports the episode boundary; rewards belong to the ending step.
        """
        grid = state.grid.clone()
        old_positions = state.agents.pos
        directions = state.agents.direction.clone()
        inventory = state.agents.inventory.clone()

        B, H, W, _ = grid.shape
        N = old_positions.shape[1]
        device = grid.device

        batch = torch.arange(B, device=device)

        # ---------------------------------------------------------
        # MOVEMENT
        # ---------------------------------------------------------

        is_move = actions < int(Action.STAY)

        target = old_positions + self.delta[actions]

        inside = (
            (target[..., 0] >= 0)
            & (target[..., 0] < H)
            & (target[..., 1] >= 0)
            & (target[..., 1] < W)
        )

        # Clamp only so indexing never goes outside the grid.
        safe_target = target.clone()
        safe_target[..., 0].clamp_(0, H - 1)
        safe_target[..., 1].clamp_(0, W - 1)

        target_object = grid[
            batch[:, None],
            safe_target[..., 0],
            safe_target[..., 1],
            0,
        ]

        empty_floor = target_object == int(S.EMPTY)

        can_move = is_move & inside & empty_floor

        proposed_positions = torch.where(
            can_move[..., None],
            target,
            old_positions,
        )

        # Trying to move changes facing direction even if movement is blocked.
        directions = torch.where(
            is_move,
            actions,
            directions,
        )

        # ---------------------------------------------------------
        # COLLISIONS
        # ---------------------------------------------------------

        blocked = torch.zeros(
            (B, N),
            dtype=torch.bool,
            device=device,
        )

        different_agents = ~torch.eye(
            N,
            dtype=torch.bool,
            device=device,
        )[None]

        # Repeat because blocking one agent may cause another collision.
        for _ in range(N):
            current_positions = torch.where(
                blocked[..., None],
                old_positions,
                proposed_positions,
            )

            same_position = (
                current_positions[:, :, None, :]
                == current_positions[:, None, :, :]
            ).all(dim=-1)

            collisions = (
                same_position & different_agents
            ).any(dim=-1)

            blocked |= collisions

        positions = torch.where(
            blocked[..., None],
            old_positions,
            proposed_positions,
        )

        # Prevent two agents from directly exchanging positions.
        moves_into_old_position = (
            positions[:, :, None, :]
            == old_positions[:, None, :, :]
        ).all(dim=-1)

        moves_into_old_position &= different_agents

        swap_pairs = (
            moves_into_old_position
            & moves_into_old_position.transpose(1, 2)
        )

        swapped = swap_pairs.any(dim=-1)

        positions = torch.where(
            swapped[..., None],
            old_positions,
            positions,
        )

        # ---------------------------------------------------------
        # 3. INTERACTIONS
        # ---------------------------------------------------------

        reward = torch.zeros(B, dtype=torch.float32, device=device)

        shaped_rewards = torch.zeros(
            (B, N),
            dtype=torch.float32,
            device=device,
        )

        correct_delivery = torch.zeros(
            B,
            dtype=torch.bool,
            device=device,
        )

        # Used by the plate pickup shaped reward.
        inventories_before_interactions = inventory.clone()

        for agent in range(N):
            front = (
                positions[:, agent]
                + self.delta[directions[:, agent]]
            )

            front_inside = (
                (front[:, 0] >= 0)
                & (front[:, 0] < H)
                & (front[:, 1] >= 0)
                & (front[:, 1] < W)
            )

            # Safe indexing.
            front[:, 0].clamp_(0, H - 1)
            front[:, 1].clamp_(0, W - 1)

            row = front[:, 0]
            col = front[:, 1]

            object_type = grid[batch, row, col, 0]
            item = grid[batch, row, col, 1]
            timer = grid[batch, row, col, 2]
            held = inventory[:, agent].clone()

            interact = (
                actions[:, agent] == int(Action.INTERACT)
            ) & front_inside

            # Object types.
            is_plate_pile = object_type == int(S.PLATE_PILE)

            ingredient_id = (
                object_type - int(S.INGREDIENT_PILE_BASE)
            )

            is_ingredient_pile = (
                (ingredient_id >= 0)
                & (ingredient_id < self.num_ingredients)
            )

            is_pile = is_plate_pile | is_ingredient_pile
            is_pot = object_type == int(S.POT)
            is_goal = object_type == int(S.GOAL)
            is_counter = object_type == int(S.WALL)

            is_button = (
                object_type == int(S.BUTTON_RECIPE_INDICATOR)
            )

            # Inventory types.
            held_is_empty = held == int(D.EMPTY)
            held_is_plate = held == int(D.PLATE)

            held_is_ingredient = (
                (held >= int(D.BASE_INGREDIENT))
                & ((held & int(D.PLATE)) == 0)
            )

            held_is_dish = (
                held & int(D.COOKED)
            ) != 0

            cell_is_empty = item == int(D.EMPTY)

            pot_is_cooking = is_pot & (timer > 0)

            pot_is_cooked = (
                is_pot
                & ((item & int(D.COOKED)) != 0)
            )

            pot_is_idle = (
                is_pot
                & ~pot_is_cooking
                & ~pot_is_cooked
            )

            # Encoded item supplied by a pile.
            safe_ingredient_id = ingredient_id.clamp(
                0,
                self.num_ingredients - 1,
            )

            pile_ingredient = self.ingredient_values[
                safe_ingredient_id
            ]

            pile_item = torch.where(
                is_plate_pile,
                int(D.PLATE),
                torch.where(
                    is_ingredient_pile,
                    pile_ingredient,
                    0,
                ),
            )

            # The combined contents after placing or picking up.
            merged_item = item + held

            plated_recipe = (
                state.recipe
                | int(D.PLATE)
                | int(D.COOKED)
            )

            # -----------------------------------------------------
            # PICKUP
            # -----------------------------------------------------

            dish_pickup = (
                interact
                & pot_is_cooked
                & held_is_plate
            )

            pile_pickup = (
                interact
                & is_pile
                & held_is_empty
            )

            counter_pickup = (
                interact
                & is_counter
                & ~cell_is_empty
                & held_is_empty
            )

            pickup = (
                dish_pickup
                | pile_pickup
                | counter_pickup
            )

            # -----------------------------------------------------
            # PLACE INGREDIENT IN POT
            # -----------------------------------------------------

            ingredient_count = (
                (
                    item[:, None]
                    // self.ingredient_values[None]
                ) % 4
            ).sum(dim=-1)

            pot_full = ingredient_count == 3

            pot_placement = (
                interact
                & pot_is_idle
                & held_is_ingredient
                & ~pot_full
            )

            # -----------------------------------------------------
            # DROP ITEM ON COUNTER
            # -----------------------------------------------------

            counter_drop = (
                interact
                & is_counter
                & cell_is_empty
                & ~held_is_empty
            )

            drop = counter_drop | pot_placement

            # -----------------------------------------------------
            # DELIVER DISH
            # -----------------------------------------------------

            delivery = (
                interact
                & is_goal
                & held_is_dish
            )

            delivery_is_correct = (
                held == plated_recipe
            )

            new_correct_delivery = (
                delivery & delivery_is_correct
            )

            correct_delivery |= new_correct_delivery

            if self.negative_rewards:
                delivery_value = torch.where(
                    delivery_is_correct,
                    DELIVERY_REWARD,
                    -DELIVERY_REWARD,
                )
            else:
                delivery_value = torch.where(
                    delivery_is_correct,
                    DELIVERY_REWARD,
                    0,
                )

            reward += delivery * delivery_value

            # -----------------------------------------------------
            # BUTTON RECIPE INDICATOR
            # -----------------------------------------------------

            indicator_activation = (
                interact
                & is_button
                & held_is_empty
                & cell_is_empty
            )

            reward -= (
                indicator_activation.float()
                * INDICATOR_ACTIVATION_COST
            )

            # -----------------------------------------------------
            # START COOKING
            # -----------------------------------------------------

            start_cooking = (
                interact
                & pot_is_idle
                & ~cell_is_empty
                & held_is_empty
            )

            # -----------------------------------------------------
            # UPDATE CELL AND INVENTORY
            # -----------------------------------------------------

            had_effect = pickup | drop | delivery
            no_effect = ~had_effect

            new_cell_item = torch.where(
                drop,
                merged_item,
                torch.where(
                    no_effect,
                    item,
                    0,
                ),
            )

            new_inventory = torch.where(
                pickup,
                pile_item + merged_item,
                torch.where(
                    no_effect,
                    held,
                    0,
                ),
            )

            grid[batch, row, col, 1] = new_cell_item
            inventory[:, agent] = new_inventory

            new_timer = torch.where(
                start_cooking,
                self.cook_time,
                timer,
            )

            new_timer = torch.where(
                indicator_activation,
                INDICATOR_ACTIVATION_TIME,
                new_timer,
            )

            grid[batch, row, col, 2] = new_timer

            # -----------------------------------------------------
            # SHAPED REWARDS
            # -----------------------------------------------------

            ingredient_selector = held | (held << 1)

            useful_placement = (
                (item & ingredient_selector)
                < (state.recipe & ingredient_selector)
            )

            shaped_rewards[:, agent] += (
                pot_placement.float()
                * torch.where(
                    useful_placement,
                    1.0,
                    -1.0 if self.negative_rewards else 0.0,
                )
                * SHAPED_REWARDS["PLACEMENT_IN_POT"]
            )

            useful_dish_pickup = (
                merged_item == plated_recipe
            )

            shaped_rewards[:, agent] += (
                dish_pickup.float()
                * useful_dish_pickup.float()
                * SHAPED_REWARDS["DISH_PICKUP"]
            )

            useful_start = item == state.recipe

            shaped_rewards[:, agent] += (
                start_cooking.float()
                * useful_start.float()
                * SHAPED_REWARDS["POT_START_COOKING"]
            )

            plate_in_hands = (
                inventories_before_interactions
                == int(D.PLATE)
            ).sum(dim=-1)

            nonempty_pots = (
                (grid[..., 0] == int(S.POT))
                & (grid[..., 1] != 0)
            ).sum(dim=(-2, -1))

            plates_on_counters = (
                (grid[..., 0] == int(S.WALL))
                & (grid[..., 1] == int(D.PLATE))
            ).any(dim=(-2, -1))

            useful_plate_pickup = (
                plate_in_hands < nonempty_pots
            ) & ~plates_on_counters

            shaped_rewards[:, agent] += (
                pile_pickup.float()
                * is_plate_pile.float()
                * useful_plate_pickup.float()
                * SHAPED_REWARDS["PLATE_PICKUP"]
            )

        # ---------------------------------------------------------
        # 4. UPDATE TIMERS
        # ---------------------------------------------------------

        static_objects = grid[..., 0]

        pots = static_objects == int(S.POT)
        pot_cooking = pots & (grid[..., 2] > 0)

        grid[..., 2] = torch.where(
            pot_cooking,
            grid[..., 2] - 1,
            grid[..., 2],
        )

        cooking_finished = (
            pot_cooking
            & (grid[..., 2] == 0)
        )

        grid[..., 1] = (
            grid[..., 1]
            | (
                cooking_finished.long()
                * int(D.COOKED)
            )
        )

        buttons = (
            static_objects == int(S.BUTTON_RECIPE_INDICATOR)
        )

        grid[..., 2] = torch.where(
            buttons,
            (grid[..., 2] - 1).clamp_min(0),
            grid[..., 2],
        )

        # ---------------------------------------------------------
        # 5. OPTIONAL NEW RECIPE AFTER DELIVERY
        # ---------------------------------------------------------

        recipe = state.recipe.clone()

        if self.sample_recipe_on_delivery:
            recipe_idx = torch.randint(
                self.layout.recipes.shape[1],
                (B,),
                device=device,
            )

            new_recipe_ids = self.layout.recipes[
                batch,
                recipe_idx,
            ]

            sampled_recipe = self._encode_ingredients(
                new_recipe_ids
            )

            recipe = torch.where(
                correct_delivery,
                sampled_recipe,
                recipe,
            )

        # ---------------------------------------------------------
        # 6. TIME AND TERMINATION
        # ---------------------------------------------------------

        time = state.time + 1

        done = (
            state.terminal
            | (time >= self.max_steps)
        )

        next_state = State(
            agents=Agents(
                pos=positions,
                inventory=inventory,
                direction=directions,
            ),
            grid=grid,
            recipe=recipe,
            time=time,
            terminal=done.clone(),
            new_correct_delivery=correct_delivery,
        )

        # Reset only finished kitchens; the others keep their step result.
        reset_indices = done.nonzero(as_tuple=True)[0]

        if reset_indices.numel() > 0:
            reset_state = self.reset(
                batch_indices=reset_indices,
            )

            next_state.agents.pos[reset_indices] = reset_state.agents.pos
            next_state.agents.direction[reset_indices] = reset_state.agents.direction
            next_state.agents.inventory[reset_indices] = reset_state.agents.inventory
            next_state.grid[reset_indices] = reset_state.grid
            next_state.recipe[reset_indices] = reset_state.recipe
            next_state.time[reset_indices] = 0
            next_state.terminal[reset_indices] = False
            next_state.new_correct_delivery[reset_indices] = False

        return next_state, reward, shaped_rewards, done
