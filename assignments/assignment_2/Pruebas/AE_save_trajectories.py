import json

import matplotlib.pyplot as plt
import numpy as np


JSON_FILE = "plot_results.json"
OUTPUT_FILE = "seed_trajectories_by_strategy.png"


def load_results():
    with open(JSON_FILE, "r") as file:
        return json.load(file)


def plot_seed_trajectories_by_strategy():
    results = load_results()

    strategies = [
        "mu_plus_lambda",
        "mu_comma_lambda",
        "random_search",
    ]

    colors = {
        "mu_plus_lambda": "tab:blue",
        "mu_comma_lambda": "tab:orange",
        "random_search": "tab:green",
    }

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

        color = colors[strategy]

        for result in strategy_results:
            history = result["best_history"]
            seed = result["seed"]
            generations = np.arange(1, len(history) + 1)

            axis.plot(
                generations,
                history,
                color=color,
                linewidth=2,
                alpha=0.75,
                label=f"seed {seed}",
            )

            axis.scatter(
                generations[0],
                history[0],
                color=color,
                edgecolor="black",
                marker="o",
                s=45,
                zorder=3,
            )

            axis.scatter(
                generations[-1],
                history[-1],
                color=color,
                edgecolor="black",
                marker="X",
                s=70,
                zorder=3,
            )

        axis.set_title(strategy)
        axis.set_xlabel("generation")
        axis.grid(True, alpha=0.4)

        if len(strategy_results) > 0:
            axis.legend(fontsize=8)

    axes[0].set_ylabel("best fitness so far")

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=200)
    plt.close()

    print(f"Saved plot: {OUTPUT_FILE}")


if __name__ == "__main__":
    plot_seed_trajectories_by_strategy()