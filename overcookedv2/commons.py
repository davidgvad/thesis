from dataclasses import dataclass
from enum import IntEnum

import torch


class Action(IntEnum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    STAY = 4
    INTERACT = 5


# Positions use (row, column).
ACTION_TO_DIR = torch.tensor(
    [
        [-1,  0],  # MOVE_UP
        [ 1,  0],  # MOVE_DOWN
        [ 0, -1],  # MOVE_LEFT
        [ 0,  1],  # MOVE_RIGHT
        [ 0,  0],  # STAY
        [ 0,  0],  # INTERACT
    ],
    dtype=torch.long,
)


class StaticObject(IntEnum):
    EMPTY = 0
    WALL = 1

    # Used only when constructing observations.
    AGENT = 2
    SELF_AGENT = 3

    GOAL = 4
    POT = 5
    RECIPE_INDICATOR = 6
    BUTTON_RECIPE_INDICATOR = 7

    PLATE_PILE = 9
    INGREDIENT_PILE_BASE = 10

    @staticmethod
    def ingredient_pile(ingredient_id):
        return int(StaticObject.INGREDIENT_PILE_BASE) + ingredient_id

    @staticmethod
    def is_ingredient_pile(value):
        return value >= int(StaticObject.INGREDIENT_PILE_BASE)

    @staticmethod
    def get_ingredient_id(pile_value):
        return pile_value - int(StaticObject.INGREDIENT_PILE_BASE)


class DynamicObject(IntEnum):
    EMPTY = 0

    # Lowest two bits store item flags.
    PLATE = 1       # binary: 01
    COOKED = 2      # binary: 10

    # Ingredient 0 starts at the third bit.
    BASE_INGREDIENT = 4  # binary: 100

    @staticmethod
    def ingredient(ingredient_id):
        """Encode one ingredient."""
        return int(DynamicObject.BASE_INGREDIENT) << (2 * ingredient_id)

    @staticmethod
    def is_ingredient(value):
        has_ingredient = (value >> 2) != 0
        has_plate = (value & int(DynamicObject.PLATE)) != 0
        return has_ingredient & ~has_plate

    @staticmethod
    def add_cooked(value):
        """Add the cooked flag."""
        return value | int(DynamicObject.COOKED)

    @staticmethod
    def add_plate(value):
        """Add the plate flag."""
        return value | int(DynamicObject.PLATE)

    @staticmethod
    def ingredient_count(value, ingredient_values):
        """
        Count ingredients in encoded values.

        value can be a batched tensor.
        ingredient_values is [4, 16, 64, ...].
        """
        counts = (
            value[..., None]
            // ingredient_values
        ) % 4

        return counts.sum(dim=-1)

    @staticmethod
    def get_ingredients_list(value):
        """
        Convert one encoded Python integer into ingredient IDs.

        This is useful for debugging, not batched environment steps.
        """
        value = int(value) >> 2

        ingredient_id = 0
        ingredients = []

        while value > 0:
            count = value & 3

            for _ in range(count):
                ingredients.append(ingredient_id)

            ingredient_id += 1
            value >>= 2

        return ingredients


@dataclass
class Agents:
    pos: torch.Tensor        # [B, N, 2], row and column
    inventory: torch.Tensor  # [B, N]
    direction: torch.Tensor  # [B, N]