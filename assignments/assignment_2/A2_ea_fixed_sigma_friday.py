import random

import mujoco as mj
import numpy as np

import A2_template_2026 as base

from ariel import console
from ariel.ec import Individual, Population, set_seed
from ariel.utils.runners import simple_runner
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder


# -----------------------------
# Parameters
# -----------------------------

POPULATION_SIZE = 6
OFFSPRING_SIZE = 6
MAX_GENERATIONS = 10

TOURNAMENT_SIZE = 2

CROSSOVER_RATE = 0.5
MUTATION_RATE = 0.1
MUTATION_SIGMA = 0.1

SIM_DURATION = 5.0

SEEDS = [0, 1]

STRATEGIES = [
    "mu_plus_lambda",
    "mu_comma_lambda",
]


# -----------------------------
# Simulation
# -----------------------------

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
    genes = np.array(genotype)

    cut = input_size * base.HIDDEN_SIZE

    w1 = genes[:cut].reshape(input_size, base.HIDDEN_SIZE)
    w2 = genes[cut:].reshape(base.HIDDEN_SIZE, output_size)

    return [w1, w2]


# -----------------------------
# Individuals
# -----------------------------

def make_individual(genotype_length):
    individual = Individual()
    individual.genotype = np.random.normal(0, 0.5, genotype_length).tolist()
    individual.requires_eval = True

    return individual


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
    simple_runner(model, data, duration=SIM_DURATION)
    mj.set_mjcb_control(None)

    final_position = base.get_core_position(data)

    fitness = base.fitness_function(initial_position, final_position)

    individual.fitness = fitness
    individual.requires_eval = False

    return fitness


def evaluate_population(population):
    for individual in population:
        if individual.requires_eval:
            evaluate_individual(individual)

    return population


# -----------------------------
# EA operators
# -----------------------------

def tournament_selection(population):
    candidates = random.choices(list(population), k=TOURNAMENT_SIZE)
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


def mutate(genotype):
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

    child_genotype = mutate(child_genotype)

    child = Individual()
    child.genotype = child_genotype
    child.requires_eval = True

    return child


def make_offspring(parents):
    offspring = Population([])

    for _ in range(OFFSPRING_SIZE):
        parent_1 = tournament_selection(parents)
        parent_2 = tournament_selection(parents)

        child = make_child(parent_1, parent_2)
        offspring.append(child)

    return offspring


def survivor_selection(parents, offspring, strategy):
    if strategy == "mu_plus_lambda":
        candidates = list(parents) + list(offspring)

    elif strategy == "mu_comma_lambda":
        candidates = list(offspring)

    else:
        raise ValueError("Unknown strategy")

    candidates.sort(key=lambda individual: individual.fitness)

    return Population(candidates[:POPULATION_SIZE])


# -----------------------------
# Video
# -----------------------------

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
        duration=SIM_DURATION,
        video_recorder=recorder,
    )

    mj.set_mjcb_control(None)


# -----------------------------
# Main EA
# -----------------------------

def run_ea(strategy, seed):
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)

    genotype_length = get_genotype_length()

    parents = Population([])

    for _ in range(POPULATION_SIZE):
        parents.append(make_individual(genotype_length))

    parents = evaluate_population(parents)

    best_history = []

    for generation in range(MAX_GENERATIONS):
        offspring = make_offspring(parents)
        offspring = evaluate_population(offspring)

        parents = survivor_selection(parents, offspring, strategy)

        best = min(parents, key=lambda individual: individual.fitness)
        best_history.append(best.fitness)

        console.log(
            f"{strategy} | seed {seed} | generation {generation + 1} | "
            f"best fitness {best.fitness:.4f}"
        )

    best_individual = min(parents, key=lambda individual: individual.fitness)

    return {
        "strategy": strategy,
        "seed": seed,
        "best_fitness": best_individual.fitness,
        "best_history": best_history,
        "best_individual": best_individual,
    }


def main():
    all_results = []

    for strategy in STRATEGIES:
        strategy_results = []

        for seed in SEEDS:
            result = run_ea(strategy, seed)

            strategy_results.append(result)
            all_results.append(result)

        best_result = min(
            strategy_results,
            key=lambda result: result["best_fitness"],
        )

        console.log(
            f"Best result for {strategy}: "
            f"{best_result['best_fitness']:.4f}"
        )

        save_video(
            best_result["best_individual"],
            strategy,
        )

    return all_results


if __name__ == "__main__":
    main()