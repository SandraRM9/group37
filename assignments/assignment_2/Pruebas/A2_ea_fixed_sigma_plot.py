import argparse
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


# ------------------------------------------------------------
# EA parameters
# ------------------------------------------------------------

POPULATION_SIZE = 10
OFFSPRING_SIZE = 10

TOURNAMENT_SIZE = 3

CROSSOVER_RATE = 0.5
MUTATION_RATE = 0.10
MUTATION_SIGMA = 0.10

MAX_GENERATIONS = 120
PLATEAU_PATIENCE = 20
PLATEAU_MIN_IMPROVEMENT = 0.001

TARGET_THRESHOLD = 0.20

SEEDS = [0, 1, 2, 3, 4]
STRATEGIES = ["mu_plus_lambda", "mu_comma_lambda"]


# ------------------------------------------------------------
# Simulation and genotype
# ------------------------------------------------------------

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

    return input_size * base.HIDDEN_SIZE + base.HIDDEN_SIZE * output_size


def genotype_to_weights(genotype, input_size, output_size):
    genes = np.asarray(genotype, dtype=np.float64)

    cut = input_size * base.HIDDEN_SIZE

    w1 = genes[:cut].reshape(input_size, base.HIDDEN_SIZE)
    w2 = genes[cut:].reshape(base.HIDDEN_SIZE, output_size)

    return [w1, w2]


# ------------------------------------------------------------
# Create and evaluate individuals
# ------------------------------------------------------------

def make_individual(genotype_length):
    individual = Individual()

    individual.genotype = np.random.normal(0, 0.5, genotype_length).tolist()
    individual.requires_eval = True

    return individual


def evaluate_individual(individual):
    model, data = make_simulation()

    input_size = len(data.qpos)
    output_size = model.nu

    weights = genotype_to_weights(individual.genotype, input_size, output_size)

    def control_callback(m, d):
        actions = base.nn_controller(m, d, weights)
        d.ctrl[:] = actions

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
    for individual in population:
        if individual.requires_eval:
            evaluate_individual(individual)

    return population


# ------------------------------------------------------------
# Parent selection and variation
# ------------------------------------------------------------

def tournament_selection(population):
    competitors = random.choices(list(population), k=TOURNAMENT_SIZE)

    best = competitors[0]

    for individual in competitors:
        if individual.fitness < best.fitness:
            best = individual

    return best


def crossover(parent_1, parent_2):
    if random.random() > CROSSOVER_RATE:
        return list(parent_1.genotype)

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


def make_child(population):
    parent_1 = tournament_selection(population)
    parent_2 = tournament_selection(population)

    child_genotype = crossover(parent_1, parent_2)
    child_genotype = gaussian_mutation(child_genotype)

    child = Individual()
    child.genotype = child_genotype
    child.requires_eval = True

    return child


def make_offspring(population):
    offspring = Population([])

    for _ in range(OFFSPRING_SIZE):
        child = make_child(population)
        offspring.append(child)

    return offspring


# ------------------------------------------------------------
# Survivor selection and stopping
# ------------------------------------------------------------

def survivor_selection(parents, offspring, strategy):
    if strategy == "mu_plus_lambda":
        candidates = list(parents) + list(offspring)

    elif strategy == "mu_comma_lambda":
        candidates = list(offspring)

    else:
        raise ValueError("Unknown strategy")

    candidates.sort(key=lambda individual: individual.fitness)

    survivors = Population([])

    for individual in candidates[:POPULATION_SIZE]:
        survivors.append(individual)

    return survivors


def has_plateau(best_history):
    if len(best_history) < PLATEAU_PATIENCE + 1:
        return False

    old_best = best_history[-PLATEAU_PATIENCE - 1]
    recent_best = min(best_history[-PLATEAU_PATIENCE:])

    improvement = old_best - recent_best

    return improvement < PLATEAU_MIN_IMPROVEMENT


def population_diversity(population):
    distances = []

    for i in range(len(population)):
        genotype_i = np.asarray(population[i].genotype)

        for j in range(i + 1, len(population)):
            genotype_j = np.asarray(population[j].genotype)

            distance = np.linalg.norm(genotype_i - genotype_j)
            distances.append(distance)

    if len(distances) == 0:
        return 0.0

    return float(np.mean(distances))


# ------------------------------------------------------------
# Run one EA
# ------------------------------------------------------------

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

    best_so_far = float("inf")
    best_individual = None

    stopping_reason = "max_generations"

    for generation in range(MAX_GENERATIONS):
        offspring = make_offspring(parents)
        offspring = evaluate_population(offspring)

        parents = survivor_selection(parents, offspring, strategy)

        fitness_values = []

        for individual in parents:
            fitness_values.append(individual.fitness)

            if individual.fitness < best_so_far:
                best_so_far = individual.fitness
                best_individual = individual

        best_now = min(fitness_values)
        mean_fitness = float(np.mean(fitness_values))
        diversity = population_diversity(parents)

        best_history.append(best_so_far)
        mean_history.append(mean_fitness)
        diversity_history.append(diversity)

        console.log(
            f"{strategy} | seed {seed} | gen {generation + 1} | "
            f"best now {best_now:.4f} | "
            f"best so far {best_so_far:.4f} | "
            f"mean {mean_fitness:.4f} | "
            f"diversity {diversity:.4f}"
        )

        if best_so_far <= TARGET_THRESHOLD:
            stopping_reason = "target_reached"
            break

        if has_plateau(best_history):
            stopping_reason = "plateau"
            break

    return {
        "strategy": strategy,
        "seed": seed,
        "best_fitness": best_so_far,
        "generations_run": len(best_history),
        "stopping_reason": stopping_reason,
        "final_diversity": diversity_history[-1],
        "best_history": best_history,
        "mean_history": mean_history,
        "diversity_history": diversity_history,
        "best_individual": best_individual,
    }


# ------------------------------------------------------------
# Trajectory and video
# ------------------------------------------------------------

def get_trajectory(individual):
    model, data = make_simulation()

    input_size = len(data.qpos)
    output_size = model.nu

    weights = genotype_to_weights(individual.genotype, input_size, output_size)

    def control_callback(m, d):
        actions = base.nn_controller(m, d, weights)
        d.ctrl[:] = actions

    trajectory = []

    mj.set_mjcb_control(control_callback)

    steps = int(base.SIM_DURATION / model.opt.timestep)

    for _ in range(steps):
        mj.mj_step(model, data)

        position = base.get_core_position(data)
        trajectory.append(position)

    mj.set_mjcb_control(None)

    return np.asarray(trajectory)


def save_trajectory_plot(individual, name):
    trajectory = get_trajectory(individual)

    plt.figure()

    plt.plot(trajectory[:, 0], trajectory[:, 1], label="robot path")
    plt.scatter(base.SPAWN_POS[0], base.SPAWN_POS[1], label="start")
    plt.scatter(base.TARGET_POSITION[0], base.TARGET_POSITION[1], label="target")

    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.title(name)
    plt.legend()
    plt.axis("equal")
    plt.grid(True)

    filename = f"{name}_trajectory.png"
    plt.savefig(filename)
    plt.close()


def save_video(individual, name):
    model, data = make_simulation()

    input_size = len(data.qpos)
    output_size = model.nu

    weights = genotype_to_weights(individual.genotype, input_size, output_size)

    def control_callback(m, d):
        actions = base.nn_controller(m, d, weights)
        d.ctrl[:] = actions

    mj.set_mjcb_control(control_callback)

    recorder = VideoRecorder(output_folder=f"__videos__/{name}")

    video_renderer(
        model,
        data,
        duration=base.SIM_DURATION,
        video_recorder=recorder,
    )

    mj.set_mjcb_control(None)


# ------------------------------------------------------------
# Fitness plots
# ------------------------------------------------------------

def pad_history(history, length):
    padded = list(history)

    while len(padded) < length:
        padded.append(padded[-1])

    return padded


def save_fitness_plot(results, strategy):
    strategy_results = []

    for result in results:
        if result["strategy"] == strategy:
            strategy_results.append(result)

    max_length = max(result["generations_run"] for result in strategy_results)

    histories = []

    for result in strategy_results:
        history = pad_history(result["best_history"], max_length)
        histories.append(history)

    histories = np.asarray(histories)

    mean_history = np.mean(histories, axis=0)
    std_history = np.std(histories, axis=0)

    generations = np.arange(1, max_length + 1)

    plt.figure()

    plt.plot(generations, mean_history, label="mean best fitness")
    plt.fill_between(
        generations,
        mean_history - std_history,
        mean_history + std_history,
        alpha=0.2,
        label="std",
    )

    plt.xlabel("generation")
    plt.ylabel("best fitness so far")
    plt.title(strategy)
    plt.legend()
    plt.grid(True)

    filename = f"{strategy}_fitness.png"
    plt.savefig(filename)
    plt.close()


# ------------------------------------------------------------
# Compare strategies
# ------------------------------------------------------------

def compare_strategies(seeds, strategies):
    all_results = []

    for strategy in strategies:
        strategy_results = []

        for seed in seeds:
            result = run_ea(strategy, seed)

            strategy_results.append(result)
            all_results.append(result)

        best_values = []
        diversity_values = []
        generation_values = []

        for result in strategy_results:
            best_values.append(result["best_fitness"])
            diversity_values.append(result["final_diversity"])
            generation_values.append(result["generations_run"])

        console.rule(f"Summary: {strategy}")
        console.log(f"Mean best fitness: {np.mean(best_values):.4f}")
        console.log(f"Std best fitness : {np.std(best_values):.4f}")
        console.log(f"Best fitness     : {np.min(best_values):.4f}")
        console.log(f"Mean generations : {np.mean(generation_values):.2f}")
        console.log(f"Mean diversity   : {np.mean(diversity_values):.4f}")

        best_result = min(strategy_results, key=lambda result: result["best_fitness"])

        save_trajectory_plot(
            best_result["best_individual"],
            f"{strategy}_best_seed_{best_result['seed']}",
        )

        save_video(
            best_result["best_individual"],
            f"{strategy}_best_seed_{best_result['seed']}",
        )

        save_fitness_plot(all_results, strategy)

    return all_results


# ------------------------------------------------------------
# Command-line arguments
    #Just to be able to run the script with different seeds and strategies, for example:
    #python A2_ea_fixed_sigma_plot.py --seed 0 --strategy mu_plus_lambda
# ------------------------------------------------------------

def read_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--strategy", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = read_arguments()

    if args.seed is None:
        seeds = SEEDS
    else:
        seeds = [args.seed]

    if args.strategy is None:
        strategies = STRATEGIES
    else:
        strategies = [args.strategy]

    compare_strategies(seeds, strategies)