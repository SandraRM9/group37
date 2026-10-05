import argparse
import random

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np

import A2_template_2026 as base

from ariel import console
from ariel.ec import Individual, Population, set_seed
from ariel.utils.renderers import video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder


# ------------------------------------------------------------
# EA parameters
# ------------------------------------------------------------

POPULATION_SIZE = 30
OFFSPRING_SIZE = 40

TOURNAMENT_SIZE = 3 #In the tournament selection, how many individuals compete to be a parent (They are selected by random).

CROSSOVER_RATE = 0.5
MUTATION_RATE = 0.10
MUTATION_SIGMA = 0.10

MAX_GENERATIONS = 100 #We select a maximum number, although in every run it stops before getting 100 generations. 
#This was more for the implementation of the code and to try its functionality.
#  The stopping criteria is the target threshold or the plateau, which are explained below.
PLATEAU_PATIENCE = 20
PLATEAU_MIN_IMPROVEMENT = 0.001

TARGET_THRESHOLD = 0.50 #We need to change to something smaller than 0.05 or maybe 0

SEEDS = [0, 1, 2, 3, 4]
STRATEGIES = ["mu_plus_lambda", "mu_comma_lambda"]


# ------------------------------------------------------------
# Simulation and genotype
# ------------------------------------------------------------

#Create and compile the MuJoCo model once
def make_model():

    mj.set_mjcb_control(None)
    world = base.build_world()
    robot = base.build_robot()

    world.spawn(
        robot.spec,
        position=base.SPAWN_POS,
        correct_collision_with_floor=True,
    )

    model = world.spec.compile()

    return model

#Create fresh simulation data for one evaluation.
def make_data(model):
   
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    return data

#Get the input and output sizes of the neural controller.
def get_network_sizes(model):

    data = make_data(model)
    input_size = len(data.qpos)
    output_size = model.nu

    return input_size, output_size


def get_genotype_length(input_size, output_size):
    """Calculate how many weights the neural network needs."""

    return input_size * base.HIDDEN_SIZE + base.HIDDEN_SIZE * output_size


def genotype_to_weights(genotype, input_size, output_size):
    """Convert one flat genotype vector into the two NN weight matrices."""

    genes = np.asarray(genotype, dtype=np.float64)

    cut = input_size * base.HIDDEN_SIZE

    w1 = genes[:cut].reshape(input_size, base.HIDDEN_SIZE)
    w2 = genes[cut:].reshape(base.HIDDEN_SIZE, output_size)

    return [w1, w2]


# ------------------------------------------------------------
# Create and evaluate individuals
# ------------------------------------------------------------

def make_individual(genotype_length):
    """Create one random individual."""

    individual = Individual()

    individual.genotype = np.random.normal(0, 0.5, genotype_length).tolist()
    individual.requires_eval = True

    return individual


def evaluate_individual(individual, model, input_size, output_size):
    """Evaluate one individual by running the robot simulation."""

    data = make_data(model)

    weights = genotype_to_weights(
        individual.genotype,
        input_size,
        output_size,
    )

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


def evaluate_population(population, model, input_size, output_size):
    """Evaluate all individuals that still need evaluation."""

    for individual in population:
        if individual.requires_eval:
            evaluate_individual(
                individual,
                model,
                input_size,
                output_size,
            )

    return population


# ------------------------------------------------------------
# Parent selection and variation
# ------------------------------------------------------------

def tournament_selection(population):
    """Select one parent using tournament selection."""

    competitors = random.choices(list(population), k=TOURNAMENT_SIZE)

    best = competitors[0]

    for individual in competitors:
        if individual.fitness < best.fitness:
            best = individual

    return best


def crossover(parent_1, parent_2):
    """Create a child genotype using arithmetic crossover."""

    if random.random() > CROSSOVER_RATE:
        return list(parent_1.genotype)

    genes_1 = np.asarray(parent_1.genotype, dtype=np.float64)
    genes_2 = np.asarray(parent_2.genotype, dtype=np.float64)

    alpha = np.random.random(genes_1.shape)

    child = alpha * genes_1 + (1 - alpha) * genes_2

    return child.tolist()


def gaussian_mutation(genotype):
    """Mutate genes using fixed-sigma Gaussian mutation."""

    genotype = np.asarray(genotype, dtype=np.float64)

    mutation_mask = np.random.random(genotype.shape) < MUTATION_RATE
    noise = np.random.normal(0, MUTATION_SIGMA, genotype.shape)

    genotype = genotype + mutation_mask * noise
    genotype = np.clip(genotype, -5.0, 5.0)

    return genotype.tolist()


def make_child(population):
    """Create one child from two selected parents."""

    parent_1 = tournament_selection(population)
    parent_2 = tournament_selection(population)

    child_genotype = crossover(parent_1, parent_2)
    child_genotype = gaussian_mutation(child_genotype)

    child = Individual()
    child.genotype = child_genotype
    child.requires_eval = True

    return child


def make_offspring(population):
    """Create the offspring population."""

    offspring = Population([])

    for _ in range(OFFSPRING_SIZE):
        child = make_child(population)
        offspring.append(child)

    return offspring


# ------------------------------------------------------------
# Survivor selection and stopping
# ------------------------------------------------------------

def survivor_selection(parents, offspring, strategy):
    """Select the next generation."""

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
    """Check whether the best fitness has stopped improving."""

    if len(best_history) < PLATEAU_PATIENCE + 1:
        return False

    old_best = best_history[-PLATEAU_PATIENCE - 1]
    recent_best = min(best_history[-PLATEAU_PATIENCE:])

    improvement = old_best - recent_best

    return improvement < PLATEAU_MIN_IMPROVEMENT


def population_diversity(population):
    """Measure average distance between genotypes."""

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
    """Run one complete evolutionary algorithm experiment."""

    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)

    model = make_model()

    input_size, output_size = get_network_sizes(model)
    genotype_length = get_genotype_length(input_size, output_size)

    parents = Population([])

    for _ in range(POPULATION_SIZE):
        individual = make_individual(genotype_length)
        parents.append(individual)

    parents = evaluate_population(
        parents,
        model,
        input_size,
        output_size,
    )

    best_history = []
    mean_history = []
    diversity_history = []

    best_so_far = float("inf")
    best_individual = None

    stopping_reason = "max_generations"

    for generation in range(MAX_GENERATIONS):
        offspring = make_offspring(parents)

        offspring = evaluate_population(
            offspring,
            model,
            input_size,
            output_size,
        )

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
    """Get the path followed by one individual."""

    model = make_model()
    data = make_data(model)

    input_size, output_size = get_network_sizes(model)

    weights = genotype_to_weights(
        individual.genotype,
        input_size,
        output_size,
    )

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
    """Save a plot of the robot trajectory."""

    trajectory = get_trajectory(individual)

    plt.figure()

    plt.plot(trajectory[:, 0], trajectory[:, 1], label="robot path")
    plt.scatter(base.SPAWN_POS[0], base.SPAWN_POS[1], label="start")
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
    plt.grid(True)

    filename = f"{name}_trajectory.png"

    plt.savefig(filename)
    plt.close()


def save_video(individual, name):
    """Save a video of the best robot."""

    model = make_model()
    data = make_data(model)

    input_size, output_size = get_network_sizes(model)

    weights = genotype_to_weights(
        individual.genotype,
        input_size,
        output_size,
    )

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
    """Extend shorter histories so all seeds have the same plot length."""

    padded = list(history)

    while len(padded) < length:
        padded.append(padded[-1])

    return padded

#Save one plot comparing both strategies with mean and std.
def save_combined_fitness_plot(results):

    plt.figure()

    colors = {
        "mu_plus_lambda": "tab:blue",
        "mu_comma_lambda": "tab:orange",
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

        console.log(
            f"{strategy} | runs: {len(strategy_results)} | "
            f"mean std: {np.mean(std_history):.6f} | "
            f"max std: {np.max(std_history):.6f}"
        )

    plt.xlabel("generation")
    plt.ylabel("best fitness so far")
    plt.title("Comparison of survivor selection strategies")
    plt.legend()
    plt.grid(True)

    filename = "strategy_comparison_fitness.png"

    plt.savefig(filename)
    plt.close()


# ------------------------------------------------------------
# Compare strategies
# ------------------------------------------------------------

def compare_strategies(seeds, strategies):
    """Run all seeds and strategies, then save plots and videos."""

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

        best_result = min(
            strategy_results,
            key=lambda result: result["best_fitness"],
        )

        save_trajectory_plot(
            best_result["best_individual"],
            f"{strategy}_best_seed_{best_result['seed']}_c",  #I'm going to add a suffix for every running 
        )

        save_video(
            best_result["best_individual"],
            f"{strategy}_best_seed_{best_result['seed']}_c",
        )

        save_fitness_plot(all_results)

    return all_results


# ------------------------------------------------------------
# Command-line arguments
# ------------------------------------------------------------

def read_arguments():
    """Read optional seed and strategy from the terminal."""

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


if __name__ == "__main__":
    compare_strategies(seeds, strategies)