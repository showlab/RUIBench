"""The tasks' and suites' names as the paper gives them.

A task card's own ``name`` (``ruibench/tasks/cards``) is part of the frozen
benchmark and stays; these are the names people read -- the command line's
listings, reports and documentation.  The agent is never told either: it gets
the task's id, level and instruction.
"""

from __future__ import annotations

from typing import Dict

#: suite prefix -> the paper's name
SUITES: Dict[str, str] = {"WK": "World knowledge", "CL": "In-context learning", "DS": "Discovery",
                          "DI": "Dynamic interaction", "CR": "Contact-rich manipulation"}

#: task id -> the paper's name
TASKS: Dict[str, str] = {
    "WK1": "Tower of Hanoi", "WK2": "River Crossing", "WK3": "Domino Run", "WK4": "Set the Table",
    "WK5": "Balance Beam", "WK6": "Tic-Tac-Toe",
    "CL1": "Play It Back", "CL2": "Dolls' Names", "CL3": "Build the House", "CL4": "Crow and Pitcher",
    "CL5": "Tangram", "CL6": "Knight's Move",
    "DS1": "Whose Present?", "DS2": "Egg Hunt", "DS3": "Treasure Hunt", "DS4": "Fake Coin",
    "DS5": "Find the Acid", "DS6": "Flame Test",
    "DI1": "Rocket Launch", "DI2": "Load the Train", "DI3": "Traffic Light", "DI4": "Simon Says",
    "DI5": "Whack-a-Mole", "DI6": "Dessert Covers",
    "CR1": "Shape Sorter", "CR2": "Screw the Bulb", "CR3": "Curling", "CR4": "Graphics Card",
    "CR5": "Snap the Bricks", "CR6": "Fold the T-Shirt",
}


def task_name(task_id: str, card_name: str = "") -> str:
    """The paper's name of a task, else its card's."""
    return TASKS.get(task_id, card_name or task_id)


def suite_name(prefix: str, card_name: str = "") -> str:
    return SUITES.get(prefix, card_name or prefix)
