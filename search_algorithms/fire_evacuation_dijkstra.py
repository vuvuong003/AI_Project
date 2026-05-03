"""
dijkstra_search.py — Dijkstra Pathfinding on the Wildfire Grid
===============================================================
Uses the EXACT same GridWorld state space as scenario.py / fire_environment.py.

Key differences from A*:
  - NO heuristic             → explores outward in all directions uniformly
  - NO fire proximity penalty → edge cost is always 1.0 (pure hop count / distance)
  - Fire cells are still IMPASSABLE (treated as walls at query time)
  - Graph structure, neighbors, and cell types are identical to GridWorld

The edge cost here is simply:
  1.0   if the destination is ROAD or EXIT
  inf   if the destination is FIRE or BLOCKED
"""

import heapq
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fire_environment import GridWorld, ROAD, FIRE, EXIT, BLOCKED
from scenario import make_custom_scenario


# ── Pure distance edge cost (no fire penalty) ──────────────────────────────────

def edge_cost_distance_only(world, to_pos):
    """
    Uniform cost of 1.0 per step.
    Fire and blocked cells are still impassable (return inf).
    No fire proximity penalty is applied — pure distance optimisation.
    """
    r, c = to_pos
    cell = world.grid[r, c]

    if cell == FIRE or cell == BLOCKED:
        return float('inf')

    return 1.0   # every passable step costs the same


# ── Dijkstra's algorithm ───────────────────────────────────────────────────────

def dijkstra_search(world):
    """
    Find the shortest path (fewest steps) from the agent to the nearest exit
    using Dijkstra's algorithm with uniform edge weights.

    Parameters
    ----------
    world : GridWorld
        The environment built by make_custom_scenario() — state space is unchanged.

    Returns
    -------
    path : list of (row, col) or None
        Ordered list of cells from start to goal, or None if unreachable.
    nodes_expanded : int
        Number of nodes popped from the priority queue.
    total_cost : float
        Total path cost (equal to path length since all edges cost 1.0).
    """
    start = world.agent_pos
    goals = set(world.exit_cells)

    # Priority queue entries: (cost_so_far, (row, col))
    open_list = []
    heapq.heappush(open_list, (0.0, start))

    came_from      = {}              # node → predecessor
    dist           = {start: 0.0}   # best known cost to each node
    nodes_expanded = 0

    while open_list:
        cost, current = heapq.heappop(open_list)

        # Skip stale entries (a shorter path was already found)
        if cost > dist.get(current, float('inf')):
            continue

        nodes_expanded += 1

        # ── Goal check ────────────────────────────────────────────────────────
        if current in goals:
            path = _reconstruct_path(came_from, current)
            return path, nodes_expanded, cost

        # ── Expand neighbours ─────────────────────────────────────────────────
        r, c = current
        for neighbor in world.get_neighbors(r, c):
            step_cost = edge_cost_distance_only(world, neighbor)
            if step_cost == float('inf'):
                continue                        # skip impassable cells

            new_cost = cost + step_cost
            if new_cost < dist.get(neighbor, float('inf')):
                dist[neighbor]      = new_cost
                came_from[neighbor] = current
                heapq.heappush(open_list, (new_cost, neighbor))

    return None, nodes_expanded, float('inf')   # no path found


# ── Path reconstruction ────────────────────────────────────────────────────────

def _reconstruct_path(came_from, current):
    """Walk back through came_from to build the path from start to goal."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


# ── Main: run Dijkstra standalone ─────────────────────────────────────────────

if __name__ == "__main__":
    world = make_custom_scenario()

    print("=" * 55)
    print("  Wildfire Evacuation — Dijkstra Search")
    print("=" * 55)
    print(f"  Grid       : {world.rows}x{world.cols}")
    print(f"  Agent at   : {world.agent_pos}")
    print(f"  Exits at   : {world.exit_cells}")
    print(f"  Fire cells : {sorted(world.fire_cells)}")
    print("=" * 55)

    path, nodes_expanded, total_cost = dijkstra_search(world)

    if path:
        print(f"\n  Path length    : {len(path)} cells")
        print(f"  Total cost     : {total_cost}")
        print(f"  Nodes expanded : {nodes_expanded}")
        world.render(path=path, title="Dijkstra — Shortest Distance Path (no fire penalty)")
    else:
        print("\n  No path found.")