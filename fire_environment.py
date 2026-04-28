"""
The world is a 2D grid where each cell has one of four states:
  0 = ROAD      (passable)
  1 = FIRE      (deadly, impassable)
  2 = EXIT      (goal)
  3 = BLOCKED   (wall/obstacle, impassable)

The grid doubles as a graph for search algorithms:
  - Nodes  = (row, col) tuples
  - Edges  = connections between adjacent ROAD/EXIT cells
  - Weight = base cost + fire proximity penalty
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import deque

# ── Cell type constants ────────────────────────────────────────────────────────
ROAD    = 0
FIRE    = 1
EXIT    = 2
BLOCKED = 3

# ── Color map for visualization ────────────────────────────────────────────────
CELL_COLORS = {
    ROAD:    "#2B2B2B",   # dark gray
    FIRE:    "#E84C0A",   # ember orange
    EXIT:    "#5BAA6A",   # green
    BLOCKED: "#111111",   # near black
}


class GridWorld:
    """
    A 2D grid environment for wildfire evacuation.

    Parameters
    ----------
    rows : int        Number of rows in the grid
    cols : int        Number of columns in the grid
    fire_spread_prob : float   Probability that fire spreads to a neighbor each timestep
    """

    def __init__(self, rows=20, cols=20, fire_spread_prob=0.3):
        self.rows = rows
        self.cols = cols
        self.fire_spread_prob = fire_spread_prob

        # Initialize all cells as ROAD
        self.grid = np.zeros((rows, cols), dtype=int)

        self.agent_pos  = None   # (row, col)
        self.exit_cells = []     # list of (row, col)
        self.fire_cells = set()  # set of (row, col)
        self.timestep   = 0

    # ── Setup helpers ──────────────────────────────────────────────────────────

    def place_agent(self, row, col):
        """Place the agent at a starting position."""
        assert self.grid[row, col] == ROAD, "Agent must start on a ROAD cell"
        self.agent_pos = (row, col)

    def place_exit(self, row, col):
        """Mark a cell as a safe exit."""
        self.grid[row, col] = EXIT
        self.exit_cells.append((row, col))

    def place_fire(self, row, col):
        """Ignite a cell."""
        self.grid[row, col] = FIRE
        self.fire_cells.add((row, col))

    def place_wall(self, row, col):
        """Mark a cell as a permanent wall."""
        self.grid[row, col] = BLOCKED

    def place_walls_rect(self, r1, c1, r2, c2):
        """Fill a rectangle of walls from (r1,c1) to (r2,c2) inclusive."""
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                self.place_wall(r, c)

    # ── Fire spread ────────────────────────────────────────────────────────────

    def spread_fire(self):
        """
        Probabilistically spread fire to neighboring ROAD cells.
        Each neighbor of a burning cell ignites with probability fire_spread_prob.
        Called once per timestep.
        """
        newly_ignited = set()

        for (r, c) in list(self.fire_cells):
            for (nr, nc) in self._neighbors(r, c):
                if self.grid[nr, nc] == ROAD:
                    if np.random.random() < self.fire_spread_prob:
                        newly_ignited.add((nr, nc))

        for (r, c) in newly_ignited:
            self.grid[r, c] = FIRE
            self.fire_cells.add((r, c))

        self.timestep += 1
        return newly_ignited  # return newly burned cells (useful for D* Lite later)

    # ── Graph interface for search algorithms ──────────────────────────────────

    def get_neighbors(self, row, col):
        """
        Return passable neighbors of a cell — used by A*, D* Lite, BFS, etc.
        Only ROAD and EXIT cells are passable.
        """
        passable = []
        for (nr, nc) in self._neighbors(row, col):
            if self.grid[nr, nc] in (ROAD, EXIT):
                passable.append((nr, nc))
        return passable

    def edge_cost(self, from_pos, to_pos):
        """
        Cost of moving from one cell to an adjacent cell.
        Base cost = 1. Adds a penalty the closer the destination is to fire.
        Returns float('inf') if the destination is on fire or blocked.
        """
        r, c = to_pos
        cell = self.grid[r, c]

        if cell == FIRE or cell == BLOCKED:
            return float('inf')

        base_cost = 1.0
        proximity_penalty = self._fire_proximity_penalty(r, c)
        return base_cost + proximity_penalty

    def is_passable(self, row, col):
        """True if the cell can be stepped on."""
        return self.grid[row, col] in (ROAD, EXIT)

    def is_goal(self, row, col):
        """True if the cell is a safe exit."""
        return self.grid[row, col] == EXIT

    def agent_is_safe(self):
        """True if the agent has reached an exit."""
        if self.agent_pos is None:
            return False
        return self.is_goal(*self.agent_pos)

    def agent_is_dead(self):
        """True if the agent is on a fire cell."""
        if self.agent_pos is None:
            return False
        return self.grid[self.agent_pos[0], self.agent_pos[1]] == FIRE

    def move_agent(self, new_pos):
        """Move the agent to a new position."""
        assert new_pos in self.get_neighbors(*self.agent_pos) or new_pos in self.exit_cells, \
            f"Invalid move: {self.agent_pos} -> {new_pos}"
        self.agent_pos = new_pos

    # ── Heuristic for A* ───────────────────────────────────────────────────────

    def heuristic(self, pos):
        """
        Manhattan distance to the nearest exit.
        Used as the A* heuristic — admissible since diagonal moves aren't allowed.
        """
        if not self.exit_cells:
            return 0
        r, c = pos
        return min(abs(r - er) + abs(c - ec) for (er, ec) in self.exit_cells)

    # ── Visualization ──────────────────────────────────────────────────────────

    def render(self, path=None, title=None):
        """
        Draw the current grid state.

        Parameters
        ----------
        path : list of (row, col), optional   Highlight a planned path in blue
        title : str, optional                 Plot title
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        # Build RGB image from grid
        img = np.zeros((self.rows, self.cols, 3))
        color_map = {
            ROAD:    [0.17, 0.17, 0.17],
            FIRE:    [0.91, 0.30, 0.04],
            EXIT:    [0.36, 0.67, 0.41],
            BLOCKED: [0.07, 0.07, 0.07],
        }
        for r in range(self.rows):
            for c in range(self.cols):
                img[r, c] = color_map[self.grid[r, c]]

        ax.imshow(img, origin='upper')

        # Draw planned path
        if path:
            path_r = [p[0] for p in path]
            path_c = [p[1] for p in path]
            ax.plot(path_c, path_r, color='#5B8CCC', linewidth=2.5,
                    linestyle='--', label='Planned path', zorder=3)

        # Draw agent
        if self.agent_pos:
            ar, ac = self.agent_pos
            ax.plot(ac, ar, 'o', color='white', markersize=12,
                    markeredgecolor='#F5A623', markeredgewidth=2.5,
                    label='Agent', zorder=5)

        # Draw exits
        for (er, ec) in self.exit_cells:
            ax.plot(ec, er, '*', color='#5BAA6A', markersize=18,
                    markeredgecolor='white', markeredgewidth=1,
                    label='Exit', zorder=4)

        # Grid lines
        ax.set_xticks(np.arange(-0.5, self.cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.rows, 1), minor=True)
        ax.grid(which='minor', color='#333333', linewidth=0.4)
        ax.tick_params(which='both', bottom=False, left=False,
                       labelbottom=False, labelleft=False)

        # Legend (deduplicated)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(),
                  loc='upper right', facecolor='#1A1A1A',
                  labelcolor='white', fontsize=9)

        title_str = title or f"Wildfire Evacuation — Timestep {self.timestep}"
        ax.set_title(title_str, color='white', fontsize=13, pad=10)
        fig.patch.set_facecolor('#1A1A1A')
        ax.set_facecolor('#1A1A1A')

        plt.tight_layout()
        plt.show()
        return fig

    # ── Private helpers ────────────────────────────────────────────────────────

    def _neighbors(self, row, col):
        """4-directional neighbors (up, down, left, right) within bounds."""
        candidates = [(row-1, col), (row+1, col), (row, col-1), (row, col+1)]
        return [(r, c) for (r, c) in candidates
                if 0 <= r < self.rows and 0 <= c < self.cols]

    def _fire_proximity_penalty(self, row, col, max_radius=3):
        """
        Returns a cost penalty based on how close a cell is to fire.
        Cells right next to fire cost more to incentivize the agent to avoid them.
        max_radius: how far away fire still affects edge cost.
        """
        min_dist = float('inf')
        for (fr, fc) in self.fire_cells:
            dist = abs(row - fr) + abs(col - fc)
            min_dist = min(min_dist, dist)

        if min_dist == 0:
            return float('inf')
        elif min_dist <= max_radius:
            return max_radius - min_dist + 1  # closer = higher penalty
        return 0.0

if __name__ == "__main__":
    from scenario import make_custom_scenario
    from search_algorithms.a_star import a_star_search
    from search_algorithms.dstar_lite import d_star_lite_search

    world = make_custom_scenario()
    print(f"Grid size:  {world.rows}x{world.cols}")
    print(f"Agent at:   {world.agent_pos}")
    print(f"Exits at:   {world.exit_cells}")
    print(f"Fire cells: {sorted(world.fire_cells)}")
    print(f"Neighbors of agent: {world.get_neighbors(*world.agent_pos)}")
    print(f"Heuristic from agent: {world.heuristic(world.agent_pos)}")

    path_a_star, nodes_a_star = a_star_search(world)
    paths_d_star, nodes_d_star = d_star_lite_search(world)
    
    print(f"A* Search --> path length: {len(path_a_star) if path_a_star else 'None'}, nodes expanded: {nodes_a_star}")
    print(f"D* Lite Search --> path length: {len(paths_d_star) if paths_d_star else 'None'}, nodes expanded: {nodes_d_star}")

    world.render(path=path_a_star, title="A* Plan")
    world.render(path=paths_d_star, title="D* Lite Plan")
