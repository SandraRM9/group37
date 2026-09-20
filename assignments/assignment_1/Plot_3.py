# Plot the results of Assignment 1

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Folder created by A1_complete_tree.py
data_folder = Path("__data__") / "copy"

# Independent runs
seeds = [10, 20, 30, 40, 50]

# EA variants
variants = ["point", "subtree", "random"]

# Number of generations used in the experiment
n_generations = 100


def get_fitness_history(database_path):
    """Get the best fitness in each generation."""

    connection = sqlite3.connect(database_path)

    data = pd.read_sql("SELECT * FROM individual",connection)

    connection.close()

    fitness_history = []

    for generation in range(n_generations + 1):

        if generation < n_generations:
            # Individuals alive in this generation
            current_population = data[
                (data["time_of_birth"] <= generation)
                & (data["time_of_death"] > generation)
            ]

        else:
            # Final population
            current_population = data[data["alive"] == 1]

        # Lower fitness is better
        best_fitness = current_population["fitness_"].min()

        fitness_history.append(best_fitness)

    return fitness_history


# Create the figure
plt.figure(figsize=(10, 5))


for variant in variants:

    all_runs = []

    # Read the five independent runs
    for seed in seeds:

        database_path = (data_folder
            / variant
            / f"seed_{seed}"
            / "database.db"
        )

        print( f"Reading {variant}, seed {seed}" )

        fitness_history = get_fitness_history( database_path)
        all_runs.append(fitness_history)

    # Convert list to NumPy array
    all_runs = np.asarray(all_runs)

    # Average fitness across the five runs
    average_fitness = np.mean(all_runs, axis=0)

    # Standard deviation across the five runs
    std_fitness = np.std(all_runs,  axis=0)

    generations = range(len(average_fitness))

    # Plot the average
    plt.plot(generations,average_fitness,label=variant)

    # Plot average ± standard deviation
    plt.fill_between(
        generations,
        average_fitness - std_fitness,
        average_fitness + std_fitness,
        alpha=0.2,
    )


plt.xlabel("Generation")
plt.ylabel("Mean best fitness")
plt.title("Mean Fitness over generations")
plt.legend()
plt.grid()

# Save the plot
plt.savefig(
    data_folder / "fitness_comparison.png",
    dpi=300,
)

plt.show()