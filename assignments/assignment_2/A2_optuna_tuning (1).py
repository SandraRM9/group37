"""Optuna tuning layer for the A2 ARIEL EA.

This file does not define the evolutionary algorithm itself. It imports the EA
from A2_ea_gaussian_ariel.py and uses Optuna only to search over EA parameters.

Important interpretation:

- The EA minimizes fitness, where fitness is distance to the target.
- Optuna also minimizes the utility value returned by `objective`.
- If your report uses the word "maximization", then the equivalent utility is:

      utility = -mean_final_fitness

  Maximizing that utility is mathematically the same as minimizing mean fitness.
"""

from __future__ import annotations

import numpy as np
import optuna

from ariel import console

from A2_ea_gaussian_ariel import EAConfig, run_ea


TUNING_SEEDS: list[int] = [0, 1, 2, 3, 4]
N_TRIALS: int = 30


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function.

    One trial represents one EA parameter vector.

    The value returned is:

        mean final best fitness across several independent seeds

    Since the original fitness is distance to the target, lower is better.
    Therefore, Optuna is configured with direction="minimize".
    """
    population_size = trial.suggest_int("population_size", 12, 50)

    survivor_strategy = trial.suggest_categorical(
        "survivor_strategy",
        ["mu_plus_lambda", "mu_comma_lambda"],
    )

    config_values = {
        "population_size": population_size,
        "offspring_size": trial.suggest_int(
            "offspring_size",
            population_size,
            population_size * 3,
        ),
        "tournament_size": trial.suggest_int(
            "tournament_size",
            2,
            min(8, population_size),
        ),
        "crossover_rate": trial.suggest_float("crossover_rate", 0.0, 1.0),
        "mutation_rate": trial.suggest_float("mutation_rate", 0.01, 0.4),
        "mutation_sigma": trial.suggest_float(
            "mutation_sigma",
            0.01,
            1.0,
            log=True,
        ),
        "survivor_strategy": survivor_strategy,
    }

    seed_results = []

    for seed in TUNING_SEEDS:
        config = EAConfig(
            seed=seed,
            max_generations=80,
            plateau_patience=10,
            plateau_min_improvement=0.01,
            verbose=False,
            **config_values,
        )

        result = run_ea(config)
        seed_results.append(result)

    final_fitness_values = [
        float(result["best_fitness"])
        for result in seed_results
    ]
    generations_to_plateau = [
        float(result["generations_to_plateau"])
        for result in seed_results
    ]
    final_diversities = [
        float(result["final_diversity"])
        for result in seed_results
    ]

    mean_final_fitness = float(np.mean(final_fitness_values))
    std_final_fitness = float(np.std(final_fitness_values))
    mean_generations_to_plateau = float(np.mean(generations_to_plateau))
    mean_final_diversity = float(np.mean(final_diversities))

    trial.set_user_attr("mean_final_fitness", mean_final_fitness)
    trial.set_user_attr("std_final_fitness", std_final_fitness)
    trial.set_user_attr(
        "mean_generations_to_plateau",
        mean_generations_to_plateau,
    )
    trial.set_user_attr("mean_final_diversity", mean_final_diversity)
    trial.set_user_attr("seed_fitness_values", final_fitness_values)

    return mean_final_fitness


def run_tuning() -> optuna.Study:
    """Run Optuna over the EA parameter space."""
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)

    console.rule("Best tuning result")
    console.log(f"Best mean final fitness: {study.best_value:.4f}")
    console.log(f"Best parameters: {study.best_params}")
    console.log(f"Analysis metrics: {study.best_trial.user_attrs}")

    return study


def main() -> None:
    run_tuning()


if __name__ == "__main__":
    main()
