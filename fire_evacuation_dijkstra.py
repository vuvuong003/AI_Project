"""
dijkstra_search.py — Dijkstra Pathfinding on the Wildfire Grid
===============================================================
Uses the EXACT same GridWorld state space as scenario.py / fire_environment.py.

Key differences from A* in fire_environment.py:
  - NO heuristic             → explores outward in all directions uniformly
  - NO fire proximity penalty → edge cost is always 1.0 (pure hop count / distance)
  - Fire cells are still IMPASSABLE (treated as walls at query time)
  - Graph structure, neighbors, and cell types are identical to GridWorld

The edge cost here is simply:
  1.0   if the destination is ROAD or EXIT
  inf   if the destination is FIRE or BLOCKED

Run this file directly to compare Dijkstra vs A* on the custom scenario.
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
        Number of nodes popped from the priority queue (for comparison with A*).
    total_cost : float
        Total path cost (equal to path length since all edges cost 1.0).
    """
    start = world.agent_pos
    goals = set(world.exit_cells)

    # Priority queue entries: (cost_so_far, (row, col))
    open_list = []
    heapq.heappush(open_list, (0.0, start))

    came_from   = {}              # node → predecessor
    dist        = {start: 0.0}   # best known cost to each node
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
                dist[neighbor]    = new_cost
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


# ── Main: run and compare with A* ─────────────────────────────────────────────

if __name__ == "__main__":
    from fire_environment import a_star_search

    world_dijkstra = make_custom_scenario()
    world_astar    = make_custom_scenario()   # same layout, separate instance

    print("=" * 55)
    print("  Wildfire Evacuation — Algorithm Comparison")
    print("=" * 55)
    print(f"  Grid       : {world_dijkstra.rows}x{world_dijkstra.cols}")
    print(f"  Agent at   : {world_dijkstra.agent_pos}")
    print(f"  Exits at   : {world_dijkstra.exit_cells}")
    print(f"  Fire cells : {sorted(world_dijkstra.fire_cells)}")
    print("=" * 55)

    # ── Run Dijkstra ──────────────────────────────────────────────────────────
    d_path, d_nodes, d_cost = dijkstra_search(world_dijkstra)

    print("\n[Dijkstra — distance only, no fire penalty]")
    if d_path:
        print(f"  Path length    : {len(d_path)} cells")
        print(f"  Total cost     : {d_cost}")
        print(f"  Nodes expanded : {d_nodes}")
    else:
        print("  No path found.")

    # ── Run A* ────────────────────────────────────────────────────────────────
    a_path, a_nodes = a_star_search(world_astar)

    print("\n[A* — distance + fire proximity penalty + heuristic]")
    if a_path:
        # Compute actual A* cost using the fire-aware edge cost
        a_cost = sum(
            world_astar.edge_cost(a_path[i], a_path[i + 1])
            for i in range(len(a_path) - 1)
        )
        print(f"  Path length    : {len(a_path)} cells")
        print(f"  Total cost     : {a_cost:.2f}")
        print(f"  Nodes expanded : {a_nodes}")
    else:
        print("  No path found.")

    print("\n" + "=" * 55)

    # ── Render both paths ─────────────────────────────────────────────────────
    if d_path:
        world_dijkstra.render(
            path=d_path,
            title="Dijkstra — Shortest Distance Path (no fire penalty)"
        )

    if a_path:
        world_astar.render(
            path=a_path,
            title="A* — Fire-Aware Least-Cost Path"
        )