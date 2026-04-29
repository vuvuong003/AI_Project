import heapq
import itertools
import math
from collections import defaultdict

def d_star_lite_search(world, start=None, goal=None):
    if start is None:
        start = world.agent_pos

    if goal is None:
        if not world.exit_cells:
            return None, 0
        goals = list(world.exit_cells)
    else:
        goals = [goal]

    best_path = None
    best_cost = math.inf
    total_nodes_expanded = 0

    for current_goal in goals:
        path, nodes = _d_star_lite_search_for_goal(world, start, current_goal)
        total_nodes_expanded += nodes
        if path is None:
            continue

        cost = sum(world.edge_cost(path[i], path[i + 1])
                   for i in range(len(path) - 1))
        if cost < best_cost:
            best_cost = cost
            best_path = path

    return best_path, total_nodes_expanded

def _d_star_lite_search_for_goal(world, start, goal):
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def calculate_key(u):
        return (min(g[u], rhs[u]) + heuristic(start, u) + km,
                min(g[u], rhs[u]))

    def top_key():
        while open_list:
            key, _, u = open_list[0]
            if u is REMOVED:
                heapq.heappop(open_list)
                continue
            return key
        return (math.inf, math.inf)

    def add_to_open(u):
        if u in entry_finder:
            remove_from_open(u)
        entry = [calculate_key(u), next(counter), u]
        entry_finder[u] = entry
        heapq.heappush(open_list, entry)

    def remove_from_open(u):
        entry = entry_finder.pop(u, None)
        if entry is not None:
            entry[2] = REMOVED

    def pop_best():
        while open_list:
            key, _, u = heapq.heappop(open_list)
            if u is REMOVED:
                continue
            entry_finder.pop(u, None)
            return key, u
        return (math.inf, math.inf), None

    def update_vertex(u):
        if u != goal:
            rhs[u] = min(
                world.edge_cost(u, succ) + g[succ]
                for succ in world.get_neighbors(*u)
            )
        if u in entry_finder:
            remove_from_open(u)
        if g[u] != rhs[u]:
            add_to_open(u)

    def compute_shortest_path():
        nonlocal nodes_expanded
        while top_key() < calculate_key(start) or rhs[start] != g[start]:
            k_old, u = pop_best()
            if u is None:
                break
            k_new = calculate_key(u)
            if k_old < k_new:
                add_to_open(u)
            elif g[u] > rhs[u]:
                g[u] = rhs[u]
                for p in world.get_neighbors(*u):
                    update_vertex(p)
            else:
                g[u] = math.inf
                update_vertex(u)
                for p in world.get_neighbors(*u):
                    update_vertex(p)
            nodes_expanded += 1

    def reconstruct_path():
        if g[start] == math.inf:
            return None
        path = [start]
        current = start
        while current != goal:
            successors = world.get_neighbors(*current)
            if not successors:
                return None
            current = min(
                successors,
                key=lambda s: world.edge_cost(current, s) + g[s]
            )
            if current in path:
                return None
            path.append(current)
        return path

    g = defaultdict(lambda: math.inf)
    rhs = defaultdict(lambda: math.inf)
    rhs[goal] = 0

    open_list = []
    entry_finder = {}
    REMOVED = "<removed>"
    counter = itertools.count()
    km = 0
    nodes_expanded = 0

    add_to_open(goal)
    compute_shortest_path()

    return reconstruct_path(), nodes_expanded