import heapq
from collections import deque

def a_star_search(world):
    start = world.agent_pos
    goals = set(world.exit_cells)

    open_list = []
    heapq.heappush(open_list, (0, start))

    came_from = {}
    g_score = {start: 0}

    nodes_expanded = 0   

    while open_list:
        _, current = heapq.heappop(open_list)
        total_cost = 0
        nodes_expanded += 1   

        # found goal?
        if current in goals:
            path = reconstruct_path(came_from, current)
            return path, nodes_expanded  

        r, c = current
        for neighbor in world.get_neighbors(r, c):
            new_g = g_score[current] + world.edge_cost(current, neighbor)
            if neighbor not in g_score or new_g < g_score[neighbor]:
                g_score[neighbor] = new_g
                f_score = new_g + world.heuristic(neighbor)

                heapq.heappush(open_list, (f_score, neighbor))
                came_from[neighbor] = current

    return None, nodes_expanded  # no path found


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path