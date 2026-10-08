from dataclasses import dataclass

import torch
from commons import StaticObject as S


@dataclass
class Layout:
    grid: torch.Tensor             # [B, H, W]
    agent_positions: torch.Tensor # [B, N, 2], (row, column)
    recipes: torch.Tensor          # [B, R, 3]

    @classmethod
    def from_string(cls, inp):
        # Convert map symbols to static grid values.
        lookup = torch.zeros(256, dtype=torch.long)

        symbols = {
            "W": S.WALL,
            "X": S.GOAL,
            "P": S.POT,
            "B": S.PLATE_PILE,
            "R": S.RECIPE_INDICATOR,
            "L": S.BUTTON_RECIPE_INDICATOR,
        }

        for symbol, value in symbols.items():
            lookup[ord(symbol)] = int(value)

        char_maps = []
        recipes = []

        # Python loops only over maps, not over their cells.
        for grid_text, possible_recipes in inp:
            rows = grid_text.split("\n")
            height, width = len(rows), len(rows[0])

            # All maps must have the same size to stack into one tensor
            if any(len(row) != width for row in rows):
                raise ValueError("All rows in a map must have equal width.")

            char_map = torch.tensor(
                [[ord(cell) for cell in row] for row in rows],
            ).reshape(height, width)

            char_maps.append(char_map)
            recipe_tensor = torch.tensor(possible_recipes, dtype=torch.long)
            if recipe_tensor.ndim != 2 or recipe_tensor.shape[1] != 3 or recipe_tensor.shape[0] == 0:
                raise ValueError("Provide at least one recipe with exactly three ingredients.")
            if ((recipe_tensor < 0) | (recipe_tensor > 9)).any():
                raise ValueError("Ingredient IDs must be between 0 and 9.")
            recipes.append(recipe_tensor)

        chars = torch.stack(char_maps).long()  # [B, H, W]

        # Map all static symbols at once.
        grid = lookup[chars]

        # Replace ingredient pile symbols with their encoded values.
        is_ingredient = (chars >= ord("0")) & (chars <= ord("9"))
        ingredient_ids = chars - ord("0")
        grid = torch.where(
            is_ingredient,
            S.INGREDIENT_PILE_BASE + ingredient_ids,
            grid,
        )

        # Find all agents at once. nonzero returns [batch, row, column].
        agent_coords = (chars == ord("A")).nonzero()
        batch_size = len(inp)
        agent_counts = (chars == ord("A")).sum(dim=(1, 2))
        if not (agent_counts == agent_counts[0]).all():
            raise ValueError("All maps in a batch must have the same number of agents.")
        num_agents = int(agent_counts[0].item())
        agent_positions = agent_coords[:, 1:].reshape(batch_size, num_agents, 2)

        return cls(
            grid=grid,
            agent_positions=agent_positions,
            recipes=torch.stack(recipes),
        )
