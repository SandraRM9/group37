import json
import random

import matplotlib.pyplot as plt
import numpy as np

import AE_ea_fixed_sigma_plot_c as ea

from ariel import console
from ariel.ec import Population, set_seed


SEEDS = [0]
JSON_FILE = "plot_results.json"
PLOT_FILE = "strategy_comparison_with_random_baseline.png"


def load_results():
    try:
        with open(JSON_FILE, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return []


def save_result(new_result):
    results = load_results()

    results = [
        result for result in results
        if not (
            result["strategy"] == new_result["strategy"]
            and result["seed"] == new_result["seed"]
        )
    ]

    results.append(new_result)

    with open(JSON_FILE, "w") as file:
        json.dump(results, file, indent=4)


def run_random_search(seed):
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)

    model = ea.make_model()
    input_size, output_size = ea.get_network_sizes(model)
    genotype_length = ea.get_genotype_length(input_size, output_size)

    best_so_far = float("inf")
    best_history = []
    diversity_history = []
    stopping_reason = "max_generations"

    for generation in range(ea.MAX_GENERATIONS):
        population = Population([])

        for _ in range(ea.POPULATION_SIZE):
            population.append(ea.make_individual(genotype_length))

        ea.evaluate_population(population, model, input_size, output_size)

        fitness_values = [individual.fitness for individual in population]
        best_so_far = min(best_so_far, min(fitness_values))

        best_history.append(best_so_far)
        diversity_history.append(ea.population_diversity(population))

        console.log(
            f"random_search | seed {seed} | gen {generation + 1} | "
            f"best so far {best_so_far:.4f}"
        )

        if best_so_far <= ea.TARGET_THRESHOLD:
            stopping_reason = "target_reached"
            break

        if ea.has_plateau(best_history):
            stopping_reason = "plateau"
            break

    return {
        "strategy": "random_search",
        "seed": seed,
        "generations_run": len(best_history),
        "best_history": best_history,
        "best_fitness": best_so_far,
        "final_diversity": diversity_history[-1],
        "stopping_reason": stopping_reason,
    }


def pad_history(history, length):
    return history + [history[-1]] * (length - len(history))


def plot_results():
    results = load_results()
    strategies = sorted(set(result["strategy"] for result in results))

    colors = {
        "mu_plus_lambda": "tab:blue",
        "mu_comma_lambda": "tab:orange",
        "random_search": "tab:green",
    }

    plt.figure(figsize=(10, 6))

    for strategy in strategies:
        strategy_results = [
            result for result in results
            if result["strategy"] == strategy
        ]

        max_length = max(result["generations_run"] for result in strategy_results)

        histories = [
            pad_history(result["best_history"], max_length)
            for result in strategy_results
        ]

        histories = np.asarray(histories)

        mean_history = np.mean(histories, axis=0)
        std_history = np.std(histories, axis=0)
        generations = np.arange(1, max_length + 1)

        color = colors.get(strategy)

        plt.plot(generations, mean_history,label=f"{strategy} mean", color=color)

        plt.fill_between(
            generations,
            mean_history - std_history,
            mean_history + std_history,
            alpha=0.25,
            color=color,
            label=f"{strategy} std",
        )

    plt.xlabel("generation")
    plt.ylabel("best fitness so far")
    plt.title("Comparison of survivor selection strategies ")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    plt.close()

    console.log(f"Saved plot: {PLOT_FILE}")


def main():
    for seed in SEEDS:
        result = run_random_search(seed)
        save_result(result)

    plot_results()


if __name__ == "__main__":
    main()