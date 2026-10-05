import json

import matplotlib.pyplot as plt
import numpy as np


JSON_FILE = "plot_results.json"
OUTPUT_FILE = "seed_trajectories_by_strategy_colored.png"


def load_results():
    with open(JSON_FILE, "r") as file:
        return json.load(file)


def get_colors(strategy, n):
    if strategy == "mu_plus_lambda":
        cmap = plt.cm.Blues
    elif strategy == "mu_comma_lambda":
        cmap = plt.cm.OrRd
    elif strategy == "random_search":
        cmap = plt.cm.Greens
    else:
        cmap = plt.cm.Greys

    if n == 1:
        return [cmap(0.75)]

    return [cmap(value) for value in np.linspace(0.45, 0.95, n)]


def plot_seed_trajectories_by_strategy():
    results = load_results()

    strategies = [
        "mu_plus_lambda",
        "mu_comma_lambda",
        "random_search",
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
        sharey=True,
    )

    for axis, strategy in zip(axes, strategies):
        strategy_results = [
            result for result in results
            if result["strategy"] == strategy
        ]

        strategy_results = sorted(
            strategy_results,
            key=lambda result: result["seed"],
        )

        colors = get_colors(strategy, len(strategy_results))

        for result, color in zip(strategy_results, colors):
            history = result["best_history"]
            seed = result["seed"]
            generations = np.arange(1, len(history) + 1)

            axis.plot(
                generations,
                history,
                color=color,
                linewidth=2.2,
                alpha=0.95,
                label=f"seed {seed}",
            )

            axis.scatter(
                generations[0],
                history[0],
                color=color,
                edgecolor="black",
                marker="o",
                s=50,
                zorder=3,
            )

            axis.scatter(
                generations[-1],
                history[-1],
                color=color,
                edgecolor="black",
                marker="X",
                s=75,
                zorder=3,
            )

        axis.set_title(strategy)
        axis.set_xlabel("generation")
        axis.grid(True, alpha=0.35)

        if len(strategy_results) > 0:
            axis.legend(fontsize=8)

    axes[0].set_ylabel("best fitness so far")

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=200)
    plt.close()

    print(f"Saved plot: {OUTPUT_FILE}")


if __name__ == "__main__":
    plot_seed_trajectories_by_strategy()