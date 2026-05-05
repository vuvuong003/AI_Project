import time
import statistics

from grids.small_grid import make_custom_scenario as make_small_scenario
from grids.medium_grid import make_custom_scenario as make_medium_scenario
from grids.large_grid import make_custom_scenario as make_large_scenario
from search_algorithms.a_star import a_star_search
from search_algorithms.dstar_lite import d_star_lite_search
from search_algorithms.fire_evacuation_dijkstra import dijkstra_search


def unpack_search_result(result):
    if result is None:
        return None, 0, 0.0
    if len(result) == 3:
        return result[0], result[1], result[2]
    return result[0], result[1], 0.0


def path_cost(world, path):
    if not path or len(path) < 2:
        return float('inf') if not path else 0.0
    return sum(world.edge_cost(path[i], path[i + 1]) for i in range(len(path) - 1))


def run_search(world, search_fn):
    start_time = time.perf_counter()
    result = search_fn(world)
    elapsed = time.perf_counter() - start_time

    path, nodes, extra_cost = unpack_search_result(result)
    if extra_cost:
        cost = extra_cost
    else:
        cost = path_cost(world, path)

    return path, nodes, cost, elapsed


def simulate_agent(world, search_fn, max_steps=500):
    total_nodes = 0
    total_time = 0.0
    replans = 0
    travel_cost = 0.0

    while not world.agent_is_safe() and max_steps > 0:
        path, nodes, _, elapsed = run_search(world, search_fn)
        total_nodes += nodes
        total_time += elapsed
        replans += 1

        if path is None or len(path) <= 1:
            return False, total_nodes, total_time, replans, travel_cost

        next_pos = path[1]
        travel_cost += world.edge_cost(world.agent_pos, next_pos)
        world.move_agent(next_pos)

        if world.agent_is_safe():
            break

        world.spread_fire()
        if world.agent_is_dead():
            return False, total_nodes, total_time, replans, travel_cost

        max_steps -= 1

    return world.agent_is_safe(), total_nodes, total_time, replans, travel_cost


def benchmark_search_method(name, search_fn, scenario_fn=make_small_scenario, trials=100, fire_spread_prob=0.1):
    dynamic_successes = 0
    dynamic_nodes = []
    dynamic_times = []
    dynamic_replans = []
    dynamic_travel_costs = []

    for _ in range(trials):
        world = scenario_fn()
        world.fire_spread_prob = fire_spread_prob
        success, nodes_dyn, time_dyn, replans, travel_cost = simulate_agent(world, search_fn)
        if success:
            dynamic_successes += 1
            dynamic_travel_costs.append(travel_cost)
        dynamic_nodes.append(nodes_dyn)
        dynamic_times.append(time_dyn)
        dynamic_replans.append(replans)

    total_replans = sum(dynamic_replans)
    avg_replan_time = total_replans and sum(dynamic_times) / total_replans

    return {
        "name": name,
        "trials": trials,
        "dynamic_success_rate": dynamic_successes / trials,
        "dynamic_avg_total_nodes": statistics.mean(dynamic_nodes),
        "dynamic_avg_total_time": statistics.mean(dynamic_times),
        "dynamic_avg_replans": statistics.mean(dynamic_replans),
        "dynamic_avg_replan_time": avg_replan_time or 0.0,
        "dynamic_avg_travel_cost": statistics.mean(dynamic_travel_costs) if dynamic_travel_costs else float('inf'),
    }


def print_summary(results):
    print("\nBenchmark summary")
    print("-----------------")
    for result in results:
        print(f"\n{result['name']}")
        print(f"  Trials                       : {result['trials']}")
        print(f"  Dynamic success rate         : {result['dynamic_success_rate']:.2%}")
        print(f"  Dynamic avg total nodes      : {result['dynamic_avg_total_nodes']:.1f}")
        print(f"  Dynamic avg total time       : {result['dynamic_avg_total_time']:.6f} sec")
        print(f"  Dynamic avg replans          : {result['dynamic_avg_replans']:.2f}")
        print(f"  Dynamic avg time per replan  : {result['dynamic_avg_replan_time']:.6f} sec")
        print(f"  Dynamic avg travel cost      : {result['dynamic_avg_travel_cost']:.3f}")


def main():
    trials = 100
    fire_spread_prob = 0.1

    print("Running benchmark for search methods...")
    print(f"Fire spread probability: {fire_spread_prob}\n")

    results = [
        benchmark_search_method("Dijkstra", dijkstra_search, trials=trials, fire_spread_prob=fire_spread_prob),
        benchmark_search_method("A*", a_star_search, trials=trials, fire_spread_prob=fire_spread_prob),
        benchmark_search_method("D* Lite", d_star_lite_search, trials=trials, fire_spread_prob=fire_spread_prob),
    ]
    print_summary(results)


def benchmark_scenarios():
    """Benchmark different scenarios with the same search algorithm"""
    trials = 50  # Fewer trials for scenario comparison
    fire_spread_prob = 0.1

    scenarios = [
        ("Small (10x10)", make_small_scenario),
        ("Medium (20x20)", make_medium_scenario),
        ("Large (30x30)", make_large_scenario),
    ]

    search_fn = a_star_search  # Use A* for comparison

    print("Comparing scenarios with A* search...")
    print(f"Fire spread probability: {fire_spread_prob}\n")

    for scenario_name, scenario_fn, trials in scenarios:
        print(f"\n--- {scenario_name} ---")
        result = benchmark_search_method(
            f"A* on {scenario_name}",
            search_fn,
            scenario_fn=scenario_fn,
            trials=trials,
            fire_spread_prob=fire_spread_prob
        )

        print(f"  Success rate: {result['dynamic_success_rate']:.2%}")
        print(f"  Avg nodes: {result['dynamic_avg_total_nodes']:.1f}")
        print(f"  Avg time: {result['dynamic_avg_total_time']:.6f} sec")
        print(f"  Avg replans: {result['dynamic_avg_replans']:.2f}")


def benchmark_all_algorithms_and_scenarios():
    """Comprehensive benchmark showing detailed results for each grid size"""
    fire_spread_prob = 0.1

    algorithms = [
        ("Dijkstra", dijkstra_search),
        ("A*", a_star_search),
        ("D* Lite", d_star_lite_search),
    ]

    scenarios = [
        ("10x10 Grid", make_small_scenario, 100),  # More trials for small grids
        ("20x20 Grid", make_medium_scenario, 50),   # Medium trials for medium grids
        ("50x50 Grid", make_large_scenario, 20),    # Fewer trials for large grids
    ]

    print("Comprehensive Algorithm Benchmark Results")
    print("=" * 80)
    print(f"Fire spread probability: {fire_spread_prob}")
    print("Trials adjusted by grid size for reasonable runtime")
    print()

    for scenario_name, scenario_fn, trials in scenarios:
        print(f"\n{scenario_name.upper()}")
        print("=" * len(scenario_name))
        print(f"Running {trials} trials per algorithm...")

        for algo_name, algo_fn in algorithms:
            result = benchmark_search_method(
                f"{algo_name} on {scenario_name}",
                algo_fn,
                scenario_fn=scenario_fn,
                trials=trials,
                fire_spread_prob=fire_spread_prob
            )

            print(f"\n{algo_name}")
            print(f"  Trials                       : {result['trials']}")
            print(f"  Dynamic success rate         : {result['dynamic_success_rate']:.2%}")
            print(f"  Dynamic avg total nodes      : {result['dynamic_avg_total_nodes']:.1f}")
            print(f"  Dynamic avg total time       : {result['dynamic_avg_total_time']:.6f} sec")
            print(f"  Dynamic avg replans          : {result['dynamic_avg_replans']:.2f}")
            print(f"  Dynamic avg time per replan  : {result['dynamic_avg_replan_time']:.6f} sec")
            print(f"  Dynamic avg travel cost      : {result['dynamic_avg_travel_cost']:.3f}")

        print("\n" + "-" * 80)


if __name__ == "__main__":
    benchmark_all_algorithms_and_scenarios()
