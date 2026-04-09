"""
scenario.py — Custom Wildfire Evacuation Map
=============================================
Recreates the hand-drawn 20x20 map exactly.
Run this file directly to visualize the map.

  W = wall (BLOCKED)
  🤖 = agent start
  🔥 = fire
  🚪 = exit
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fire_environment import GridWorld, ROAD, FIRE, EXIT, BLOCKED

def make_custom_scenario():
    """
    Build the exact map from the hand-drawn design.
    Grid is 20 columns x 20 rows (0-indexed).
    """
    world = GridWorld(rows=20, cols=20, fire_spread_prob=0.3)

    # ── WALLS (W) ─────────────────────────────────────────────────────────────
    # Read row by row from the image (row 0 = top)

    walls = [
        # Row 0 — top border (all except col 4 which is agent)
        (0,0),(0,1),(0,2),(0,3),(0,5),(0,6),(0,7),(0,8),(0,9),
        (0,10),(0,11),(0,12),(0,13),(0,14),(0,15),(0,16),(0,17),(0,18),(0,19),

        # Row 1
        (1,0),(1,19),

        # Row 2
        (2,0),(2,19),

        # Row 3
        (3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,15),(3,19),

        # Row 4
        (4,0),(4,5),(4,15),(4,19),

        # Row 5
        (5,0),(5,9),(5,15),(5,16),(5,17),(5,18),(5,19),

        # Row 6
        (6,0),(6,6),(6,7),(6,8),(6,9),(6,10),(6,11),(6,19),

        # Row 7
        (7,0),(7,11),(7,19),

        # Row 8
        (8,0),(8,11),(8,19),

        # Row 9
        (9,0),(9,4),(9,5),(9,6),(9,7),(9,8),(9,11),(9,12),(9,13),
        (9,16),(9,17),(9,18),(9,19),

        # Row 10
        (10,0),(10,8),(10,19),

        # Row 11
        (11,0),(11,8),(11,19),

        # Row 12
        (12,0),(12,10),(12,19),

        # Row 13
        (13,0),(13,10),(13,14),(13,15),(13,16),(13,17),(13,18),(13,19),

        # Row 14
        (14,0),(14,10),(14,19),

        # Row 15
        (15,0),(15,10),(15,19),

        # Row 16 — long horizontal wall
        (16,0),(16,3),(16,4),(16,5),(16,6),(16,7),(16,8),(16,9),(16,10),
        (16,11),(16,12),(16,13),(16,14),(16,15),(16,16),(16,17),(16,18),(16,19),

        # Row 17
        (17,0),(17,19),

        # Row 18
        (18,0),(18,19),

        # Row 18 — bottom border
        (19,0),(19,1),(19,2),(19,3),(19,4),(19,5),(19,6),(19,7),(19,8),
        (19,9),(19,10),(19,11),(19,12),(19,13),(19,14),(19,15),(19,16),
        (19,17),(19,18),(19,19),
    ]

    for (r, c) in walls:
        world.place_wall(r, c)

    # ── FIRE 🔥 ───────────────────────────────────────────────────────────────
    fires = [
        (2, 17),    # row 2, right area
        (5, 2),(5, 3),   # row 5, left cluster
        (6, 2),(6, 3),   # row 6, left cluster
        (10, 5),(10, 6), # row 10, mid-left cluster
        (11, 12),(11, 13), # row 11, mid-right cluster
        (14, 7),    # row 14, center
    ]

    for (r, c) in fires:
        world.place_fire(r, c)

    # ── EXITS 🚪 ──────────────────────────────────────────────────────────────
    world.place_exit(14, 19)   # right side, row 14
    world.place_exit(19, 12)   # bottom area, row 18 (gap in bottom wall)

    # ── AGENT 🤖 ──────────────────────────────────────────────────────────────
    world.place_agent(0, 4)    # row 0, col 4

    return world


if __name__ == "__main__":
    world = make_custom_scenario()

    print(f"Grid:     {world.rows}x{world.cols}")
    print(f"Agent:    {world.agent_pos}")
    print(f"Exits:    {world.exit_cells}")
    print(f"Fires:    {sorted(world.fire_cells)}")
    print(f"Passable neighbors of agent: {world.get_neighbors(*world.agent_pos)}")

    world.render(title="Custom Wildfire Map — Initial State")