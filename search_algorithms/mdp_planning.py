import heapq
import numpy as np
# mdp_planning.py
from environment_two import FIRE, BLOCKED

def compute_fire_risk(world, decay=3):
    risk = np.zeros((world.rows, world.cols))

    for r in range(world.rows):
        for c in range(world.cols):

            min_dist = float('inf')

            for (fr, fc) in world.fire_cells:
                dist = abs(r - fr) + abs(c - fc)
                min_dist = min(min_dist, dist)

            if min_dist == 0:
                risk[r, c] = 1.0
            else:
                risk[r, c] = np.exp(-min_dist / decay)

    return risk


def mdp_search(world):
    """
    Risk-aware A* style MDP approximation (balanced version)
    Returns: (path, nodes_expanded)
    """

    import heapq

    start = world.agent_pos
    goals = set(world.exit_cells)

    risk_map = compute_fire_risk(world, decay=6)  # smoother risk spread

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}

    nodes = 0

    def heuristic(pos):
        # distance to closest exit
        r, c = pos
        return min(abs(r - gr) + abs(c - gc) for gr, gc in goals)

    while open_set:

        _, current = heapq.heappop(open_set)
        nodes += 1

        if current in goals:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1], nodes

        for neighbor in world.get_neighbors(*current):

            r, c = neighbor
            cell = world.grid[r][c]

            if cell in (FIRE, BLOCKED):
                continue

            # 🔥 BALANCED MDP COST FUNCTION
            # (IMPORTANT: reduced alpha + smoother risk)
            step_cost = 1.0 + 1.5 * risk_map[r][c]

            # small urgency boost toward exits
            urgency = 0.2 * heuristic(neighbor)

            tentative = g_score[current] + step_cost

            if neighbor not in g_score or tentative < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative

                f = tentative + urgency
                heapq.heappush(open_set, (f, neighbor))

    return None, nodes