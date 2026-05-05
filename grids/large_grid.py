import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fire_environment import GridWorld, ROAD, FIRE, EXIT, BLOCKED

def make_custom_scenario():
    rows, cols = 50, 50
    world = GridWorld(rows=rows, cols=cols, fire_spread_prob=0.08)

    walls = []

    # 1. PERIMETER
    for r in range(rows):
        for c in range(cols):
            # Top wall (with gap for agent at col 10)
            if r == 0 and c != 10:
                walls.append((r, c))
            # Bottom wall (with gap for exit at col 25)
            if r == rows - 1 and c != 25:
                walls.append((r, c))
            # Left and Right borders
            if c == 0 or c == cols - 1:
                # Leave a gap for the side exit at row 35 and row 5
                if not (c == cols - 1 and r == 35) and not (c == cols - 1 and r == 5):
                    walls.append((r, c))

    # 2. INTERNAL HORIZONTAL BARRIERS
    # Row 7-8
    for c in range(1, 15): walls.append((7, c)); walls.append((8, c))
    for c in range(35, 49): walls.append((7, c)); walls.append((8, c))

    # Row 15 - Mid-top cluster
    for c in range(15, 30): walls.append((15, c))
    
    # Row 22-24 - The "Central Maze" divider
    for c in range(10, 20): walls.append((22, c)); walls.append((23, c))
    for c in range(28, 35): walls.append((22, c)); walls.append((23, c))
    for c in range(40, 49): walls.append((22, c)); walls.append((23, c))

    # Row 32-34 - Lower barrier
    for c in range(25, 49): 
        walls.append((32, c))
        
        walls.append((33, c))

    # Row 40 - Long horizontal barrier near bottom
    for c in range(1, 45):
        if c not in [8, 9, 10, 11, 12, 13, 14]: # hallway gap
            walls.append((40, c))

    # 3. VERTICAL CONNECTORS
    for r in range(15, 25): walls.append((r, 25)) # Mid-divider
    for r in range(25, 35): walls.append((r, 10)) # Left-side chamber

    for (r, c) in walls:
        if 0 <= r < rows and 0 <= c < cols:
            world.place_wall(r, c)

    # 4. FIRE 🔥
    fires = [
        (5, 40), (4,40), (4,41), (5,41),   # Upper right
        (25, 27),  # Center
        (28, 2),   # Lower left
        (45, 45),  # Lower right corner
        (12, 22)   # Near the start path
    ]

    for (r, c) in fires:
        world.place_fire(r, c)

    # 5. EXITS 🚪
    world.place_exit(5, 49) 
    world.place_exit(35, 49)   # Right side exit
    world.place_exit(49, 25)   # Bottom exit

    # 6. AGENT 🤖
    world.place_agent(0, 10)    # Scaled starting position

    return world


if __name__ == "__main__":
    world = make_custom_scenario()

    print(f"Grid:     {world.rows}x{world.cols}")
    print(f"Agent:    {world.agent_pos}")
    print(f"Exits:    {world.exit_cells}")
    print(f"Fires:    {len(world.fire_cells)} fire cells")
    print(f"Passable neighbors of agent: {world.get_neighbors(*world.agent_pos)}")

    world.render(title="Large Wildfire Map (50x50) — Initial State")