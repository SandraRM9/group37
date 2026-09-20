# Plot fitness and body size results for Assignment 1

import json
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ariel.ec.genotypes.tree.tree_genome import TreeGenome

from scipy.stats import wilcoxon


# Folder created by the experiment script
data_folder = Path("__data__") / "A1_complete_tree_b"

# Independent runs
seeds = [10, 20, 30, 40, 50]

# Experimental variants
variants = ["point", "subtree", "random"]

# Number of generations
n_generations = 100


def get_current_population(data, generation):
    """Return the individuals alive in a given generation."""

    if generation < n_generations:
        return data[
            (data["time_of_birth"] <= generation)
            & (data["time_of_death"] > generation)
        ]

    # At the final generation, use the individuals still alive
    return data[data["alive"] == 1]


def get_histories(database_path):
    """Get best fitness and body size of the best individual per generation."""

    if not database_path.is_file():
        raise FileNotFoundError(f"Database not found: {database_path}")

    connection = sqlite3.connect(database_path)

    try:
        data = pd.read_sql("SELECT * FROM individual", connection)
    finally:
        connection.close()

    fitness_history = []
    body_size_history = []

    for generation in range(n_generations + 1):
        current_population = get_current_population(data, generation)

        if current_population.empty:
            raise ValueError(
                f"No individuals found for generation {generation} "
                f"in {database_path}"
            )

        # Find the individual with the lowest fitness
        best_index = current_population["fitness_"].idxmin()
        best_individual = current_population.loc[best_index]

        fitness_history.append(best_individual["fitness_"])

        # Reconstruct the best individual's body
        genotype = best_individual["genotype_"]

        if isinstance(genotype, str):
            genotype = json.loads(genotype)

        genome = TreeGenome.from_dict(genotype)
        body = genome.to_networkx()

        # Body size = number of modules
        body_size_history.append(body.number_of_nodes())

    return fitness_history, body_size_history


def plot_comparison(results, metric, ylabel, title, filename):
    """Plot mean ± standard deviation across the five independent runs."""

    plt.figure(figsize=(10, 5))

    for variant in variants:
        all_runs = results[variant][metric]

        mean_values = np.mean(all_runs, axis=0)
        std_values = np.std(all_runs, axis=0)
        generations = range(len(mean_values))

        line, = plt.plot(
            generations,
            mean_values,
            label=variant,
        )

        plt.fill_between(
            generations,
            mean_values - std_values,
            mean_values + std_values,
            color=line.get_color(),
            alpha=0.2,
        )

    plt.xlabel("Generation")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()

    output_path = data_folder / filename
    plt.savefig(output_path, dpi=300)

    print(f"Saved: {output_path}")
    plt.show()


def main():
    results = {}

    # Read all five runs for each variant
    for variant in variants:
        fitness_runs = []
        body_size_runs = []

        for seed in seeds:
            database_path = (
                data_folder
                / variant
                / f"seed_{seed}"
                / "database.db"
            )

            print(f"Reading {variant}, seed {seed}")

            fitness_history, body_size_history = get_histories(database_path)

            fitness_runs.append(fitness_history)
            body_size_runs.append(body_size_history)

        results[variant] = {
            "fitness": np.asarray(fitness_runs),
            "body_size": np.asarray(body_size_runs),
        }

    # Graph 1: fitness of the best individual
    plot_comparison(
        results=results,
        metric="fitness",
        ylabel="Mean best fitness",
        title="Mean best fitness over generations",
        filename="fitness_comparison.png",
    )

    # Graph 2: body size of the best individual
    plot_comparison(
        results=results,
        metric="body_size",
        ylabel="Mean body size of best individual (modules)",
        title="Mean body size of the best individual over generations",
        filename="body_size_comparison.png",
    )

    #final best fitness point, subtree, random
    point_final_fitness = results["point"]["fitness"][:,-1]
    subtree_final_fitness = results["subtree"]["fitness"][:,-1]
    random_final_fitness = results["random"]["fitness"][:,-1]

    #mean and sd for point, subtree, random
    print(
        "Mean point:", np.mean(point_final_fitness),
        "\nSD point:", np.std(point_final_fitness),
        "\nMean subtree:", np.mean(subtree_final_fitness),
        "\nSD subtree:", np.std(subtree_final_fitness),
        "\nMean random:", np.mean(random_final_fitness),
        "\nSD random:", np.std(random_final_fitness),
    )

    #Wilcoxon test one-sided
    result_wilcoxon_point_sub = wilcoxon(point_final_fitness, subtree_final_fitness, alternative="less") #point vs subtree
    result_wilcoxon_point_random = wilcoxon(point_final_fitness, random_final_fitness, alternative="less") #point vs subtree
    result_wilcoxon_sub_random = wilcoxon(subtree_final_fitness, random_final_fitness, alternative="less") #point vs subtree

    print(
        "Wilcoxon point subtree:", result_wilcoxon_point_sub,
        "\nWilcoxon point random:", result_wilcoxon_point_random,
        "\nWilcoxon subtree random:", result_wilcoxon_sub_random
    )


if __name__ == "__main__":
    main()

