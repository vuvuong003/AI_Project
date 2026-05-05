import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fire_environment import GridWorld, ROAD, FIRE, EXIT, BLOCKED

def make_custom_scenario():
    rows, cols = 10, 10
    # Slightly lower spread prob because the grid is so small
    world = GridWorld(rows=rows, cols=cols, fire_spread_prob=0.05)

    # 1. PERIMETER
    walls = []
    for c in range(cols):
        if c != 2: walls.append((0, c))    # Top wall (gap for agent at col 2)
        if c != 2: walls.append((9, c))    # Bottom wall (gap for exit at col 2)
    
    for r in range(1, 9):
        walls.append((r, 0))               # Left wall
        walls.append((r, 9))    # Right wall (gap for exit at row 6)

    # 2. INTERNAL BARRIERS (Condensed bottlenecks)
    # Mid-top barrier
    for c in range(1, 4): walls.append((3, c))
    
    # Mid-bottom barrier except for a narrow gap at (6, 3)
    for c in range(4, 9):
        if c != 6:
            walls.append((6, c))

    for (r, c) in walls:
        if 0 <= r < rows and 0 <= c < cols:
            world.place_wall(r, c)

    # 3. FIRE 🔥
    fires = [
        (2, 7),   # Upper right pocket
        (5, 2),   # Mid-left area
        (8, 8)    # Near the bottom-right exit
    ]

    for (r, c) in fires:
        world.place_fire(r, c)

    # 4. EXITS 🚪
    world.place_exit(9, 2)   # Bottom exit

    # 5. AGENT 🤖
    world.place_agent(0, 2)  # Starting position

    return world


if __name__ == "__main__":
    world = make_custom_scenario()

    print(f"Grid:     {world.rows}x{world.cols}")
    print(f"Agent:    {world.agent_pos}")
    print(f"Exits:    {world.exit_cells}")
    print(f"Fires:    {sorted(world.fire_cells)}")
    print(f"Passable neighbors of agent: {world.get_neighbors(*world.agent_pos)}")

    world.render(title="Custom Wildfire Map — Initial State")