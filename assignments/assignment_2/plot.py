import json
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

STRATEGIES = [
    "mu_plus_lambda",
    "mu_comma_lambda",
    "random_search",
]


# ------------------------------------------------------------
# Load saved results
# ------------------------------------------------------------

def load_plot_data(filename="plot_results_b.json"):
    """Load previously saved EA results."""

    with open(filename, "r") as file:
        return json.load(file)


# ------------------------------------------------------------
# Fitness plot
# ------------------------------------------------------------

def pad_history(history, length):
    """Extend shorter histories so all seeds have the same plot length."""

    padded = list(history)

    while len(padded) < length:
        padded.append(padded[-1])

    return padded


def save_combined_fitness_plot(results):

    plt.figure()

    colors = {
        "mu_plus_lambda": "tab:blue",
        "mu_comma_lambda": "tab:orange",
        "random_search": "tab:green",
    }

    for strategy in STRATEGIES:

        strategy_results = []

        for result in results:
            if result["strategy"] == strategy:
                strategy_results.append(result)

        if len(strategy_results) == 0:
            continue

        max_length = max(
            result["generations_run"]
            for result in strategy_results
        )

        histories = []

        for result in strategy_results:
            history = pad_history(
                result["best_history"],
                max_length,
            )
            histories.append(history)

        histories = np.asarray(histories)

        mean_history = np.mean(histories, axis=0)
        std_history = np.std(histories, axis=0)

        generations = np.arange(1, max_length + 1)

        color = colors[strategy]

        plt.plot(
            generations,
            mean_history,
            label=f"{strategy} mean",
            color=color,
        )

        plt.fill_between(
            generations,
            mean_history - std_history,
            mean_history + std_history,
            color=color,
            alpha=0.25,
            label=f"{strategy} std",
        )

        print(
            f"{strategy} | runs: {len(strategy_results)} | "
            f"mean std: {np.mean(std_history):.6f} | "
            f"max std: {np.max(std_history):.6f}"
        )

    plt.xlabel("generation")
    plt.ylabel("best fitness so far")
    plt.title("Comparison of survivor selection strategies")
    plt.legend()
    plt.grid(True)

    filename = "strategy_comparison_fitness_b.png"

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":

    results = load_plot_data("plot_results_b.json")

    save_combined_fitness_plot(results)