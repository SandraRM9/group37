"""Optuna-based EA implementation for A2.

This file keeps the original A2 template mostly untouched. It imports the
world, robot, controller and fitness definitions from A2_template_2026.py, then
adds:

- a real-valued genotype representing neural-network weights,
- Gaussian mutation,
- tournament parent selection,
- two survivor-selection strategies: (mu + lambda) and (mu, lambda),
- plateau-based stopping,
- Optuna tuning over symbolic and numeric EA parameters.

Research-question fit:
How does survivor selection strategy affect convergence and diversity when
evolving neural-network controllers for locomotion?
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import mujoco as mj
import numpy as np
import numpy.typing as npt
import optuna

import A2_template_2026 as base

from ariel import console
from ariel.ec import set_seed
from ariel.utils.runners import simple_runner


SurvivorStrategy = Literal["mu_plus_lambda", "mu_comma_lambda"]


# --------------------------------------------------------------------------- #
# Experiment settings
# --------------------------------------------------------------------------- #

TUNING_SEEDS: list[int] = [0, 1, 2, 3, 4]
N_TRIALS: int = 30

# Safety limit. The main stopping criterion is plateau, but this prevents runs
# from taking forever if a plateau is not detected.
MAX_GENERATIONS: int = 80

# Plateau criterion: stop if best fitness does not improve enough for this many
# consecutive generations.
PLATEAU_PATIENCE: int = 10
PLATEAU_MIN_IMPROVEMENT: float = 0.01

# Weight values can explode after repeated mutation. Clipping keeps the MuJoCo
# controller numerically stable without changing the representation.
WEIGHT_MIN: float = -5.0
WEIGHT_MAX: float = 5.0


# --------------------------------------------------------------------------- #
# Individual representation
# --------------------------------------------------------------------------- #

@dataclass
class Individual:
    genotype: npt.NDArray[np.float64]
    fitness: float | None = None
    history: dict[str, float] = field(default_factory=dict)


def get_controller_sizes() -> tuple[int, int, int]:
    """Return input size, output size and flat genotype length."""
    mj.set_mjcb_control(None)

    world = base.build_world()
    robot = base.build_robot()

    world.spawn(
        robot.spec,
        position=base.SPAWN_POS,
        correct_collision_with_floor=True,
    )

    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    input_size = len(data.qpos)
    output_size = model.nu
    genotype_length = (
        input_size * base.HIDDEN_SIZE
        + base.HIDDEN_SIZE * output_size
    )

    return input_size, output_size, genotype_length


def genotype_to_weights(
    genotype: npt.NDArray[np.float64],
    input_size: int,
    output_size: int,
) -> list[npt.NDArray[np.float64]]:
    """Convert a flat vector of real-valued genes into [w1, w2]."""
    cut = input_size * base.HIDDEN_SIZE
    required = cut + base.HIDDEN_SIZE * output_size

    if genotype.size != required:
        raise ValueError(f"Expected {required} weights, got {genotype.size}")

    w1 = genotype[:cut].reshape(input_size, base.HIDDEN_SIZE)
    w2 = genotype[cut:].reshape(base.HIDDEN_SIZE, output_size)

    return [w1, w2]


def make_individual(
    rng: np.random.Generator,
    genotype_length: int,
) -> Individual:
    genotype = rng.normal(0.0, 0.5, genotype_length)
    return Individual(genotype=genotype)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate_individual(individual: Individual) -> float:
    """Run one headless simulation and return fitness. Lower is better."""
    mj.set_mjcb_control(None)

    world = base.build_world()
    robot = base.build_robot()

    world.spawn(
        robot.spec,
        position=base.SPAWN_POS,
        correct_collision_with_floor=True,
    )

    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    input_size = len(data.qpos)
    output_size = model.nu
    weights = genotype_to_weights(individual.genotype, input_size, output_size)

    def control_callback(m: mj.MjModel, d: mj.MjData) -> None:
        actions = base.nn_controller(m, d, weights)

        if not np.all(np.isfinite(actions)):
            d.ctrl[:] = 0.0
            return

        d.ctrl[:] = actions

    initial_position = base.get_core_position(data)
    mj.set_mjcb_control(control_callback)
    simple_runner(model, data, duration=base.SIM_DURATION)
    mj.set_mjcb_control(None)

    final_position = base.get_core_position(data)
    fitness = base.fitness_function(initial_position, final_position)

    individual.fitness = fitness
    individual.history["final_x"] = float(final_position[0])
    individual.history["final_y"] = float(final_position[1])

    return fitness


# --------------------------------------------------------------------------- #
# EA operators
# --------------------------------------------------------------------------- #

def tournament_selection(
    population: list[Individual],
    tournament_size: int,
    rng: np.random.Generator,
) -> Individual:
    """Select one parent. Lower fitness is better."""
    indices = rng.choice(len(population), size=tournament_size, replace=False)
    candidates = [population[int(i)] for i in indices]
    return min(candidates, key=lambda individual: individual.fitness)


def arithmetic_crossover(
    parent_1: Individual,
    parent_2: Individual,
    rng: np.random.Generator,
) -> Individual:
    """Create one child by averaging parent genotypes."""
    alpha = rng.random()
    child_genotype = (
        alpha * parent_1.genotype
        + (1.0 - alpha) * parent_2.genotype
    )
    return Individual(genotype=child_genotype.copy())


def gaussian_mutation(
    individual: Individual,
    mutation_rate: float,
    mutation_sigma: float,
    rng: np.random.Generator,
) -> None:
    """Mutate real-valued neural-network weights with Gaussian noise."""
    mask = rng.random(individual.genotype.size) < mutation_rate
    noise = rng.normal(0.0, mutation_sigma, individual.genotype.size)
    individual.genotype = individual.genotype + mask * noise
    individual.genotype = np.clip(individual.genotype, WEIGHT_MIN, WEIGHT_MAX)


def survivor_selection(
    parents: list[Individual],
    offspring: list[Individual],
    population_size: int,
    strategy: SurvivorStrategy,
) -> list[Individual]:
    """Apply either (mu + lambda) or (mu, lambda) survivor selection."""
    if strategy == "mu_plus_lambda":
        candidates = parents + offspring
    elif strategy == "mu_comma_lambda":
        candidates = offspring
    else:
        raise ValueError(f"Unknown survivor strategy: {strategy}")

    candidates = sorted(candidates, key=lambda individual: individual.fitness)
    return candidates[:population_size]


def population_diversity(population: list[Individual]) -> float:
    """Average distance between genotypes. Higher means more diversity."""
    if len(population) < 2:
        return 0.0

    distances: list[float] = []

    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            distance = np.linalg.norm(
                population[i].genotype - population[j].genotype
            )
            distances.append(float(distance))

    return float(np.mean(distances))


def has_plateau(
    best_fitness_history: list[float],
    patience: int = PLATEAU_PATIENCE,
    min_improvement: float = PLATEAU_MIN_IMPROVEMENT,
) -> bool:
    """Return True when the best fitness has barely improved recently."""
    if len(best_fitness_history) < patience + 1:
        return False

    old_fitness = best_fitness_history[-patience - 1]
    recent_best = min(best_fitness_history[-patience:])
    improvement = old_fitness - recent_best

    return improvement < min_improvement


# --------------------------------------------------------------------------- #
# One full EA run
# --------------------------------------------------------------------------- #

def run_ea(
    *,
    seed: int,
    population_size: int,
    offspring_size: int,
    tournament_size: int,
    mutation_rate: float,
    mutation_sigma: float,
    crossover_rate: float,
    survivor_strategy: SurvivorStrategy,
    max_generations: int = MAX_GENERATIONS,
) -> dict[str, object]:
    """Run one EA until plateau or max_generations."""
    np.random.seed(seed)
    set_seed(seed)
    rng = np.random.default_rng(seed)

    _, _, genotype_length = get_controller_sizes()

    population = [
        make_individual(rng, genotype_length)
        for _ in range(population_size)
    ]

    for individual in population:
        evaluate_individual(individual)

    best_fitness_history: list[float] = []
    mean_fitness_history: list[float] = []
    diversity_history: list[float] = []

    for generation in range(max_generations):
        offspring: list[Individual] = []

        while len(offspring) < offspring_size:
            parent_1 = tournament_selection(population, tournament_size, rng)
            parent_2 = tournament_selection(population, tournament_size, rng)

            if rng.random() < crossover_rate:
                child = arithmetic_crossover(parent_1, parent_2, rng)
            else:
                child = Individual(genotype=parent_1.genotype.copy())

            gaussian_mutation(
                child,
                mutation_rate=mutation_rate,
                mutation_sigma=mutation_sigma,
                rng=rng,
            )
            evaluate_individual(child)
            offspring.append(child)

        population = survivor_selection(
            parents=population,
            offspring=offspring,
            population_size=population_size,
            strategy=survivor_strategy,
        )

        fitness_values = [float(individual.fitness) for individual in population]
        best_fitness = min(fitness_values)
        mean_fitness = float(np.mean(fitness_values))
        diversity = population_diversity(population)

        best_fitness_history.append(best_fitness)
        mean_fitness_history.append(mean_fitness)
        diversity_history.append(diversity)

        console.log(
            f"seed={seed} gen={generation + 1} "
            f"best={best_fitness:.4f} mean={mean_fitness:.4f} "
            f"diversity={diversity:.4f}"
        )

        if has_plateau(best_fitness_history):
            break

    best_individual = min(population, key=lambda individual: individual.fitness)

    return {
        "seed": seed,
        "best_fitness": float(best_individual.fitness),
        "generations": len(best_fitness_history),
        "final_diversity": float(diversity_history[-1]),
        "best_fitness_history": best_fitness_history,
        "mean_fitness_history": mean_fitness_history,
        "diversity_history": diversity_history,
        "best_genotype": best_individual.genotype.copy(),
    }


# --------------------------------------------------------------------------- #
# Optuna objective
# --------------------------------------------------------------------------- #

def objective(trial: optuna.Trial) -> float:
    """Tune EA configuration. One trial = one parameter vector."""
    population_size = trial.suggest_int("population_size", 12, 50)

    # For (mu, lambda), lambda should be at least mu so that enough offspring
    # exist to refill the population after parents are discarded.
    offspring_size = trial.suggest_int(
        "offspring_size",
        population_size,
        population_size * 3,
    )

    tournament_size = trial.suggest_int(
        "tournament_size",
        2,
        min(8, population_size),
    )

    survivor_strategy: SurvivorStrategy = trial.suggest_categorical(
        "survivor_strategy",
        ["mu_plus_lambda", "mu_comma_lambda"],
    )

    mutation_rate = trial.suggest_float("mutation_rate", 0.01, 0.40)
    mutation_sigma = trial.suggest_float(
        "mutation_sigma",
        0.01,
        1.00,
        log=True,
    )
    crossover_rate = trial.suggest_float("crossover_rate", 0.0, 1.0)

    results = []

    for seed in TUNING_SEEDS:
        run_result = run_ea(
            seed=seed,
            population_size=population_size,
            offspring_size=offspring_size,
            tournament_size=tournament_size,
            mutation_rate=mutation_rate,
            mutation_sigma=mutation_sigma,
            crossover_rate=crossover_rate,
            survivor_strategy=survivor_strategy,
        )
        results.append(run_result)

    final_fitness_values = [
        float(result["best_fitness"])
        for result in results
    ]
    generations_to_plateau = [
        float(result["generations"])
        for result in results
    ]
    final_diversities = [
        float(result["final_diversity"])
        for result in results
    ]

    mean_fitness = float(np.mean(final_fitness_values))
    std_fitness = float(np.std(final_fitness_values))
    mean_generations = float(np.mean(generations_to_plateau))
    mean_diversity = float(np.mean(final_diversities))

    trial.set_user_attr("mean_fitness", mean_fitness)
    trial.set_user_attr("std_fitness", std_fitness)
    trial.set_user_attr("mean_generations_to_plateau", mean_generations)
    trial.set_user_attr("mean_final_diversity", mean_diversity)
    trial.set_user_attr("seed_fitness_values", final_fitness_values)

    # Fitness is still the utility used by the tuner. Convergence and diversity
    # are stored as analysis variables for the report.
    return mean_fitness


def run_optuna() -> optuna.Study:
    """Run the automated tuning procedure."""
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)

    console.rule("Best Optuna Trial")
    console.log(f"Best utility / mean fitness: {study.best_value:.4f}")
    console.log(f"Best parameters: {study.best_params}")
    console.log(f"Extra metrics: {study.best_trial.user_attrs}")

    return study


def main() -> None:
    input_size, output_size, genotype_length = get_controller_sizes()

    console.rule("Controller")
    console.log(f"controller inputs  : {input_size}")
    console.log(f"controller outputs : {output_size}")
    console.log(f"genotype length    : {genotype_length}")

    run_optuna()


if __name__ == "__main__":
    main()
