"""
Assignment 2 - EA comparing survivor selection strategies.

This code compares:
1. mu_plus_lambda: parents and offspring compete.
2. mu_comma_lambda: only offspring compete.

The neural network weights are evolved using:
- tournament selection
- intermediate recombination / arithmetic crossover
- Gaussian mutation with fixed sigma
- plateau stopping criterion
"""

import random

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np

import A2_template_2026 as base

from ariel import console
from ariel.ec import EAOperation, Individual, Population, set_seed
from ariel.utils.renderers import video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

POPULATION_SIZE = 15
OFFSPRING_SIZE = 15

TOURNAMENT_SIZE = 3

CROSSOVER_RATE = 0.5
MUTATION_RATE = 0.1
MUTATION_SIGMA = 0.1

MAX_GENERATIONS = 100

PLATEAU_PATIENCE = 15
PLATEAU_MIN_IMPROVEMENT = 0.002

TARGET_THRESHOLD = 0.20

SEEDS = [0]#1, 2, 3, 4]

STRATEGIES = [
    "mu_plus_lambda",
    "mu_comma_lambda",
]


# ---------------------------------------------------------------------------
# Simulation and neural network helpers
# ---------------------------------------------------------------------------

def make_simulation():
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

    return model, data


def get_genotype_length():
    model, data = make_simulation()

    input_size = len(data.qpos)
    output_size = model.nu

    input_to_hidden = input_size * base.HIDDEN_SIZE
    hidden_to_output = base.HIDDEN_SIZE * output_size

    return input_to_hidden + hidden_to_output


def genotype_to_weights(genotype, input_size, output_size):
    genes = np.array(genotype)

    cut = input_size * base.HIDDEN_SIZE

    w1 = genes[:cut].reshape(input_size, base.HIDDEN_SIZE)
    w2 = genes[cut:].reshape(base.HIDDEN_SIZE, output_size)

    return [w1, w2]


def make_individual(genotype_length):
    individual = Individual()

    individual.genotype = np.random.normal( 0,
        0.5,
        genotype_length,
    ).tolist()

    individual.requires_eval = True

    return individual


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_individual(individual):
    model, data = make_simulation()

    input_size = len(data.qpos)
    output_size = model.nu

    weights = genotype_to_weights(
        individual.genotype,
        input_size,
        output_size,
    )

    def controller(model, data):
        actions = base.nn_controller(model, data, weights)
        data.ctrl[:] = actions

    initial_position = base.get_core_position(data)

    mj.set_mjcb_control(controller)
    simple_runner(model, data, duration=base.SIM_DURATION)
    mj.set_mjcb_control(None)

    final_position = base.get_core_position(data)

    fitness = base.fitness_function(
        initial_position,
        final_position,
    )

    individual.fitness = fitness
    individual.requires_eval = False

    return fitness


@EAOperation
def evaluate_population(population: Population) -> Population:
    for individual in population:
        if individual.requires_eval:
            evaluate_individual(individual)

    return population


# ---------------------------------------------------------------------------
# Parent selection, crossover and mutation
# ---------------------------------------------------------------------------

def tournament_selection(population):
    candidates = random.choices(
        list(population),
        k=TOURNAMENT_SIZE,
    )

    return min(candidates, key=lambda individual: individual.fitness)


def crossover(parent_1, parent_2):
    child_genotype = []

    for i in range(len(parent_1.genotype)):
        gene_1 = parent_1.genotype[i]
        gene_2 = parent_2.genotype[i]

        alpha = random.random()

        child_gene = alpha * gene_1 + (1 - alpha) * gene_2

        child_genotype.append(child_gene)

    return child_genotype


def gaussian_mutation(genotype):
    new_genotype = []

    for gene in genotype:
        if random.random() < MUTATION_RATE:
            gene = gene + random.gauss(0, MUTATION_SIGMA)

        gene = max(-5.0, min(5.0, gene))

        new_genotype.append(gene)

    return new_genotype


def make_child(parent_1, parent_2):
    if random.random() < CROSSOVER_RATE:
        child_genotype = crossover(parent_1, parent_2)
    else:
        child_genotype = list(parent_1.genotype)

    child_genotype = gaussian_mutation(child_genotype)

    child = Individual()
    child.genotype = child_genotype
    child.requires_eval = True

    return child


def make_offspring(population):
    offspring = Population([])

    for _ in range(OFFSPRING_SIZE):
        parent_1 = tournament_selection(population)
        parent_2 = tournament_selection(population)

        child = make_child(parent_1, parent_2)

        offspring.append(child)

    return offspring


# ---------------------------------------------------------------------------
# Survivor selection
# ---------------------------------------------------------------------------

def survivor_selection(parents, offspring, strategy):
    if strategy == "mu_plus_lambda":
        candidates = list(parents) + list(offspring)

    elif strategy == "mu_comma_lambda":
        candidates = list(offspring)

    else:
        raise ValueError("Unknown strategy")

    candidates.sort(key=lambda individual: individual.fitness)

    return Population(candidates[:POPULATION_SIZE])


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def population_diversity(population):
    distances = []

    for i in range(len(population)):
        genotype_1 = np.array(population[i].genotype)

        for j in range(i + 1, len(population)):
            genotype_2 = np.array(population[j].genotype)

            distance = np.linalg.norm(genotype_1 - genotype_2)
            distances.append(distance)

    if len(distances) == 0:
        return 0.0

    return float(np.mean(distances))


def has_plateau(best_history):
    if len(best_history) < PLATEAU_PATIENCE + 1:
        return False

    old_best = best_history[-PLATEAU_PATIENCE - 1]
    recent_best = min(best_history[-PLATEAU_PATIENCE:])

    improvement = old_best - recent_best

    return improvement < PLATEAU_MIN_IMPROVEMENT


# ---------------------------------------------------------------------------
# Video and trajectory plot
# ---------------------------------------------------------------------------

def get_trajectory(individual):
    model, data = make_simulation()

    input_size = len(data.qpos)
    output_size = model.nu

    weights = genotype_to_weights(
        individual.genotype,
        input_size,
        output_size,
    )

    trajectory = []

    def controller(model, data):
        actions = base.nn_controller(model, data, weights)
        data.ctrl[:] = actions

    mj.set_mjcb_control(controller)

    steps = int(base.SIM_DURATION / model.opt.timestep)

    for _ in range(steps):
        mj.mj_step(model, data)

        position = base.get_core_position(data)
        trajectory.append(position.copy())

    mj.set_mjcb_control(None)

    return np.array(trajectory)


def save_trajectory_plot(individual, name):
    trajectory = get_trajectory(individual)

    plt.figure()

    plt.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        label="robot path",
    )

    plt.scatter(
        base.SPAWN_POS[0],
        base.SPAWN_POS[1],
        label="start",
    )

    plt.scatter(
        base.TARGET_POSITION[0],
        base.TARGET_POSITION[1],
        label="target",
    )

    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.title(name)
    plt.legend()
    plt.axis("equal")

    plt.savefig(f"{name}_trajectory.png")
    plt.close()


def save_video(individual, name):
    model, data = make_simulation()

    input_size = len(data.qpos)
    output_size = model.nu

    weights = genotype_to_weights(
        individual.genotype,
        input_size,
        output_size,
    )

    def controller(model, data):
        actions = base.nn_controller(model, data, weights)
        data.ctrl[:] = actions

    mj.set_mjcb_control(controller)

    recorder = VideoRecorder(
        output_folder=f"__videos__/{name}",
    )

    video_renderer(
        model,
        data,
        duration=base.SIM_DURATION,
        video_recorder=recorder,
    )

    mj.set_mjcb_control(None)


# ---------------------------------------------------------------------------
# Evolutionary algorithm
# ---------------------------------------------------------------------------

def run_ea(strategy, seed):
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)

    genotype_length = get_genotype_length()

    parents = Population([])

    for _ in range(POPULATION_SIZE):
        individual = make_individual(genotype_length)
        parents.append(individual)

    parents = evaluate_population(parents)

    best_history = []
    mean_history = []
    diversity_history = []

    stopping_reason = "max_generations"

    for generation in range(MAX_GENERATIONS):
        offspring = make_offspring(parents)
        offspring = evaluate_population(offspring)

        parents = survivor_selection(parents, offspring, strategy)

        fitness_values = []

        for individual in parents:
            fitness_values.append(individual.fitness)

        best_fitness = min(fitness_values)
        mean_fitness = float(np.mean(fitness_values))
        diversity = population_diversity(parents)

        best_history.append(best_fitness)
        mean_history.append(mean_fitness)
        diversity_history.append(diversity)

        console.log(
            f"{strategy} | seed {seed} | gen {generation + 1} | "
            f"best {best_fitness:.4f} | "
            f"mean {mean_fitness:.4f} | "
            f"diversity {diversity:.4f}"
        )

        if best_fitness <= TARGET_THRESHOLD:
            stopping_reason = "target_reached"
            break

        if has_plateau(best_history):
            stopping_reason = "plateau"
            break

    best_individual = min(parents, key=lambda individual: individual.fitness)

    return {
        "strategy": strategy,
        "seed": seed,
        "best_fitness": best_individual.fitness,
        "generations_run": len(best_history),
        "stopping_reason": stopping_reason,
        "final_diversity": diversity_history[-1],
        "best_history": best_history,
        "mean_history": mean_history,
        "diversity_history": diversity_history,
        "best_individual": best_individual,
    }


def compare_strategies():
    all_results = []

    for strategy in STRATEGIES:
        strategy_results = []

        for seed in SEEDS:
            result = run_ea(strategy, seed)

            strategy_results.append(result)
            all_results.append(result)

        best_values = [
            result["best_fitness"]
            for result in strategy_results
        ]

        generation_values = [
            result["generations_run"]
            for result in strategy_results
        ]

        diversity_values = [
            result["final_diversity"]
            for result in strategy_results
        ]

        best_result = min(
            strategy_results,
            key=lambda result: result["best_fitness"],
        )

        console.rule(f"Summary: {strategy}")

        console.log(f"Mean best fitness: {np.mean(best_values):.4f}")
        console.log(f"Std best fitness : {np.std(best_values):.4f}")
        console.log(f"Best fitness     : {best_result['best_fitness']:.4f}")
        console.log(f"Mean generations : {np.mean(generation_values):.2f}")
        console.log(f"Mean diversity   : {np.mean(diversity_values):.4f}")

        save_trajectory_plot(
            best_result["best_individual"],
            strategy,
        )

        save_video(
            best_result["best_individual"],
            strategy,
        )

    return all_results


if __name__ == "__main__":
    compare_strategies()