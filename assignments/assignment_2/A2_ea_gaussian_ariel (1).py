"""A2 EA implementation using ARIEL structures and Gaussian mutation.

This file keeps the original A2 template as the source of truth for:

- the robot body,
- the world,
- the neural-network controller,
- the fitness function,
- the simulation duration.

The evolutionary part uses ARIEL's `Individual`, `Population`, `EAOperation`,
and `set_seed`. The genotype is a flat vector of neural-network weights.

Main experimental question:
How does survivor selection strategy affect convergence and diversity when
evolving neural-network controllers for locomotion?
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal, cast

import mujoco as mj
import numpy as np
import numpy.typing as npt

import A2_template_2026 as base

from ariel import console
from ariel.ec import EAOperation, Individual, Population, set_seed
from ariel.utils.runners import simple_runner


SurvivorStrategy = Literal["mu_plus_lambda", "mu_comma_lambda"]


@dataclass
class EAConfig:
    seed: int = 42
    population_size: int = 20
    offspring_size: int = 20
    tournament_size: int = 3
    crossover_rate: float = 0.5
    mutation_rate: float = 0.1
    mutation_sigma: float = 0.1
    survivor_strategy: SurvivorStrategy = "mu_plus_lambda"
    max_generations: int = 80
    plateau_patience: int = 10
    plateau_min_improvement: float = 0.010
    verbose: bool = True


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)
    return np.random.default_rng(seed)


def get_controller_sizes() -> tuple[int, int, int]:
    """Return input size, output size and total genotype length."""
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
    genotype: list[float],
    input_size: int,
    output_size: int,
) -> list[npt.NDArray[np.float64]]:
    """Convert one flat genotype into the [w1, w2] matrices."""
    genes = np.asarray(genotype, dtype=np.float64)
    cut = input_size * base.HIDDEN_SIZE
    required = cut + base.HIDDEN_SIZE * output_size

    if genes.size != required:
        raise ValueError(f"Expected {required} weights, got {genes.size}")

    return [
        genes[:cut].reshape(input_size, base.HIDDEN_SIZE),
        genes[cut:].reshape(base.HIDDEN_SIZE, output_size),
    ]


def create_individual(
    rng: np.random.Generator,
    genotype_length: int,
) -> Individual:
    """Create one ARIEL Individual with a real-valued weight vector."""
    individual = Individual()
    individual.genotype = rng.normal(0.0, 0.5, genotype_length).tolist()
    individual.fitness = None
    individual.requires_eval = True
    individual.tags = {}
    return individual


def evaluate_individual(individual: Individual) -> float:
    """Evaluate one individual by running one headless MuJoCo simulation."""
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
    genotype = cast("list[float]", individual.genotype)
    weights = genotype_to_weights(genotype, input_size, output_size)

    def control_callback(m: mj.MjModel, d: mj.MjData) -> None:
        actions = base.nn_controller(m, d, weights)
        if np.all(np.isfinite(actions)):
            d.ctrl[:] = actions
        else:
            d.ctrl[:] = 0.0

    initial_position = base.get_core_position(data)
    mj.set_mjcb_control(control_callback)
    simple_runner(model, data, duration=base.SIM_DURATION)
    mj.set_mjcb_control(None)

    final_position = base.get_core_position(data)
    fitness = base.fitness_function(initial_position, final_position)

    individual.fitness = fitness
    individual.requires_eval = False
    return fitness


@EAOperation
def evaluate_population(population: Population) -> Population:
    """Evaluate every individual that still requires evaluation."""
    for individual in population:
        if individual.requires_eval:
            evaluate_individual(individual)
    return population


def tournament_pick(
    population: Population,
    tournament_size: int,
) -> Individual:
    """Pick one parent by tournament selection. Lower fitness is better."""
    competitors = random.choices(list(population), k=tournament_size)
    return min(competitors, key=lambda individual: individual.fitness)


def arithmetic_crossover(
    parent_1: Individual,
    parent_2: Individual,
) -> list[float]:
    """Real-valued crossover for neural-network weights."""
    genotype_1 = np.asarray(cast("list[float]", parent_1.genotype))
    genotype_2 = np.asarray(cast("list[float]", parent_2.genotype))
    alpha = random.random()
    child = alpha * genotype_1 + (1.0 - alpha) * genotype_2
    return child.tolist()


def gaussian_mutation(
    genotype: list[float],
    mutation_rate: float,
    mutation_sigma: float,
) -> list[float]:
    """Gaussian mutation: each selected weight receives N(0, sigma) noise."""
    genes = np.asarray(genotype, dtype=np.float64)
    mask = np.random.random(genes.size) < mutation_rate
    noise = np.random.normal(0.0, mutation_sigma, genes.size)
    mutated = genes + mask * noise
    mutated = np.clip(mutated, -5.0, 5.0)
    return mutated.tolist()


def make_child(
    parent_1: Individual,
    parent_2: Individual,
    config: EAConfig,
) -> Individual:
    """Create one child using crossover and Gaussian mutation."""
    if random.random() < config.crossover_rate:
        child_genotype = arithmetic_crossover(parent_1, parent_2)
    else:
        child_genotype = list(cast("list[float]", parent_1.genotype))

    child_genotype = gaussian_mutation(
        child_genotype,
        mutation_rate=config.mutation_rate,
        mutation_sigma=config.mutation_sigma,
    )

    child = Individual()
    child.genotype = child_genotype
    child.fitness = None
    child.requires_eval = True
    child.tags = {"offspring": True}
    return child


def make_offspring(population: Population, config: EAConfig) -> Population:
    """Create lambda offspring from the current population."""
    offspring = Population()

    for _ in range(config.offspring_size):
        parent_1 = tournament_pick(population, config.tournament_size)
        parent_2 = tournament_pick(population, config.tournament_size)
        offspring.append(make_child(parent_1, parent_2, config))

    return offspring


def survivor_selection(
    parents: Population,
    offspring: Population,
    config: EAConfig,
) -> Population:
    """Apply either (mu + lambda) or (mu, lambda) survivor selection."""
    if config.survivor_strategy == "mu_plus_lambda":
        candidates = list(parents) + list(offspring)
    elif config.survivor_strategy == "mu_comma_lambda":
        candidates = list(offspring)
    else:
        raise ValueError(f"Unknown strategy: {config.survivor_strategy}")

    candidates = sorted(candidates, key=lambda individual: individual.fitness)
    survivors = Population(candidates[: config.population_size])

    for individual in survivors:
        individual.tags = {}

    return survivors


def genotype_diversity(population: Population) -> float:
    """Average pairwise distance between genotypes."""
    if len(population) < 2:
        return 0.0

    distances: list[float] = []

    for i in range(len(population)):
        genotype_i = np.asarray(cast("list[float]", population[i].genotype))
        for j in range(i + 1, len(population)):
            genotype_j = np.asarray(cast("list[float]", population[j].genotype))
            distances.append(float(np.linalg.norm(genotype_i - genotype_j)))

    return float(np.mean(distances))


def has_plateau(
    best_fitness_history: list[float],
    patience: int,
    min_improvement: float,
) -> bool:
    """Plateau if best fitness barely improves for `patience` generations."""
    if len(best_fitness_history) < patience + 1:
        return False

    old_fitness = best_fitness_history[-patience - 1]
    recent_best = min(best_fitness_history[-patience:])
    improvement = old_fitness - recent_best

    return improvement < min_improvement


def run_ea(config: EAConfig) -> dict[str, object]:
    """Run one EA until plateau or max_generations."""
    rng = seed_everything(config.seed)
    _, _, genotype_length = get_controller_sizes()

    population = Population(
        [
            create_individual(rng, genotype_length)
            for _ in range(config.population_size)
        ],
    )
    population = evaluate_population(population)

    best_fitness_history: list[float] = []
    mean_fitness_history: list[float] = []
    diversity_history: list[float] = []

    for generation in range(config.max_generations):
        offspring = make_offspring(population, config)
        offspring = evaluate_population(offspring)
        population = survivor_selection(population, offspring, config)

        fitness_values = [float(ind.fitness) for ind in population]
        best_fitness = min(fitness_values)
        mean_fitness = float(np.mean(fitness_values))
        diversity = genotype_diversity(population)

        best_fitness_history.append(best_fitness)
        mean_fitness_history.append(mean_fitness)
        diversity_history.append(diversity)

        if config.verbose:
            console.log(
                f"gen={generation + 1} best={best_fitness:.4f} "
                f"mean={mean_fitness:.4f} diversity={diversity:.4f}",
            )

        if has_plateau(
            best_fitness_history,
            patience=config.plateau_patience,
            min_improvement=config.plateau_min_improvement,
        ):
            break

    best_individual = min(population, key=lambda individual: individual.fitness)

    return {
        "best_individual": best_individual,
        "best_fitness": float(best_individual.fitness),
        "generations_to_plateau": len(best_fitness_history),
        "final_diversity": diversity_history[-1],
        "best_fitness_history": best_fitness_history,
        "mean_fitness_history": mean_fitness_history,
        "diversity_history": diversity_history,
        "config": config,
    }


def main() -> None:
    config = EAConfig(
        seed=42,
        population_size=20,
        offspring_size=20,
        tournament_size=3,
        crossover_rate=0.5,
        mutation_rate=0.1,
        mutation_sigma=0.1,
        survivor_strategy="mu_plus_lambda",
        verbose=True,
    )

    result = run_ea(config)
    console.rule("EA result")
    console.log(f"Best fitness: {result['best_fitness']:.4f}")
    console.log(f"Generations: {result['generations_to_plateau']}")
    console.log(f"Final diversity: {result['final_diversity']:.4f}")


if __name__ == "__main__":
    main()
