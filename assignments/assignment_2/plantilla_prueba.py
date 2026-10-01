"""EC A2 template code - neuroevolution for targeted locomotion with ARIEL.

WHAT THIS FILE IS
-----------------
A *demo*, not a solution. It spawns a robot, drives it with a neural network
whose weights are RANDOM, runs the simulation, and reports how close the robot
ended up to a target.

There is deliberately NO evolution in here. Building the EA (representation,
initialisation, parent selection, variation, survivor selection) is the assignment.
See "YOUR JOB" at the bottom of this file.

THE ASSIGNMENT IN A NUTSHELL
------------------------------
Evolve the weights of a neural network controller so that a robot moves from
SPAWN_POS to TARGET_POSITION within the simulation time.

    fitness = distance between the robot's final position and TARGET_POSITION

HOW TO RUN
----------

Change MODE below to switch between an interactive viewer, a headless run,
a rendered video, or a single frame.
"""

# Standard library
from pathlib import Path
import argparse
import csv
import json
from typing import Literal

# Third-party libraries
import mujoco as mj
import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
from mujoco import viewer

# Local libraries (ARIEL)
from ariel import console
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.john_set import gecko
from ariel.ec import EA, EAOperation, Individual, Population, set_seed
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder

# Type aliases
type ViewerTypes = Literal["launcher", "video", "simple", "frame", "no_control"]

# --- RANDOM GENERATOR SETUP --- #
# Fix the seed while you are debugging.
# Report results over MULTIPLE seeds.
SEED = 42
RNG = np.random.default_rng(SEED)

# ariel.ec's own generators/mutators/crossover draw from a separate,
# package-level RNG. Reseed it too if you build your EA on ariel.ec,
# or every one of your "multiple seeds" runs the same variation operators.
set_seed(SEED)

# --- DATA SETUP --- #
SCRIPT_NAME = Path(__file__).stem
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

# --- EXPERIMENT CONSTANTS --- #
SPAWN_POS: list[float] = [0.0, 0.0, 0.1]  # where the robot starts
TARGET_POSITION: list[float] = [2.0, 0.0, 0.1]  # where it should end up
SIM_DURATION: float = 15.0  # seconds of simulated time per evaluation
MODE: ViewerTypes = "launcher"  # see run_experiment() for the options


# ============================================================================ #
#  1. THE BODY AND THE WORLD
# ============================================================================ #
def build_world() -> SimpleFlatWorld:
    """Create the environment the robot lives in.

    YOU MAY CHANGE THIS. Options include: SimpleFlatWorld, RuggedTerrainWorld,
    CraterTerrainWorld, AmphitheatreTerrainWorld, OlympicArena, ...
    (SimpleTiltedWorld is not supported for this task.)

    Whatever you pick, keep it FIXED for all runs you compare against each
    other, and say in your report which one you used. A controller evolved on
    flat ground and one evolved on rugged terrain are not comparable numbers.
    """
    return SimpleFlatWorld()


def build_robot() -> CoreModule:
    """Create the robot body.

    YOU MAY CHANGE THIS. Options include the prebuilt bodies in
    `ariel.body_phenotypes.robogen_lite.prebuilt_robots` (gecko, spider, ...).

    Two consequences of this choice, and they matter:
      * The body determines `model.nu` (the number of hinges you must send
        commands to) - that is the OUTPUT size of your controller.
      * The body determines the size of `data.qpos` - if you feed qpos to your
        network, that is (part of) your INPUT size.
    Change the body and your genotype length changes with it. Keep the body
    FIXED within an experiment.
    """
    return gecko()


# ============================================================================ #
#  2. THE CONTROLLER CONTRACT
# ============================================================================ #
#
# MuJoCo calls the controller every physics step with (model, data); its job
# is to write into data.ctrl.
#
#   INPUTS   : whatever you read from `data` (qpos, qvel, time, ...), plus any
#              task info you already know, e.g. the vector to TARGET_POSITION.
#              INPUT SIZE is your choice, but must stay CONSTANT.
#   OUTPUTS  : exactly `model.nu` values, one per actuated hinge.
#   RANGE    : hinges accept [-pi/2, +pi/2] radians. A tanh output gives
#              [-1, 1] - rescale: actions * (np.pi / 2).
#   WRITING  : DIRECT (data.ctrl[:] = actions) commands the angle straight -
#              fast, but can destabilise the sim on large jumps. DELTA
#              (data.ctrl[:] += actions * alpha, alpha ~ 0.05, then clip) is
#              smoother but accumulates, so clipping is required. Pick one,
#              justify it, use it everywhere.
#   NaN      : blown-up weights silently write NaN into data.ctrl. Assert
#              against it while developing.
#
# ============================================================================ #

# Controller architecture - decide before writing your EA.
HIDDEN_SIZE: int = 6

# ADDED: common settings for both research questions.
MU = 12
LAMBDA = 36
SIGMA = 0.10
PATIENCE = 12
MIN_GENERATIONS = 20
MAX_GENERATIONS = 100  # safety limit; report when it is reached.
IMPROVEMENT = 1e-4
DEFAULT_SEEDS = (11, 22, 33, 44, 55)
EVAL_LOGS = False



#his one we should check what is de reason for adding the sin and cos of the time, 
# but it is a way to give the network a sense of time, so it can learn to produce rhythmic patterns of movement. 
# The target direction is also added to give the network information about where the target is located relative to the robot's current position.
#  This can help the network learn to move towards the target more effectively.
def controller_inputs(data: mj.MjData) -> npt.NDArray[np.float64]:
    """ADDED: body state, rhythmic clock and direction to fixed target."""
    target_direction = (
        np.asarray(TARGET_POSITION[:2]) - np.asarray(data.qpos[:2])
    )
    return np.concatenate((
        np.asarray(data.qpos),
        [np.sin(2 * np.pi * data.time), np.cos(2 * np.pi * data.time)],
        target_direction,
    ))



def nn_controller(
    model: mj.MjModel,
    data: mj.MjData,
    weights: list[npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Map robot state to hinge commands: in -> hidden -> actions.

    In this demo `weights` is drawn at RANDOM. In your assignment, `weights`
    is what the evolutionary algorithm produces: an individual's genotype,
    reshaped into these matrices. You are free to change the architecture
    itself (layers, activations, ...) - just keep input/output sizes correct.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model. Use `model.nu` for the number of hinges.
    data : mj.MjData
        The MuJoCo data. This is where you read the robot's state from.
    weights : list of ndarray
        [w1, w2] - the layer weight matrices.

    Returns
    -------
    npt.NDArray[np.float64]
        `model.nu` action values, already scaled to [-pi/2, pi/2].
    """
    w1, w2 = weights

    # --- INPUTS ---------------------------------------------------------- #
    # Bare qpos - the simplest choice, not necessarily a good one. See
    # YOUR JOB below.

    #inputs = data.qpos --> esto era lo que habia antes, pero lo he cambiado para añadir el sin y cos del tiempo y la direccion al target
    inputs = controller_inputs(data)

    # --- FORWARD PASS ----------------------------------------------------- #
    layer1 = np.tanh(inputs @ w1)
    outputs = np.tanh(layer1 @ w2)  # in [-1, 1]

    # --- RESCALE TO THE HINGE RANGE --------------------------------------- #
    return outputs * (np.pi / 2)  # in [-pi/2, pi/2]


def make_random_weights(
    input_size: int,
    output_size: int,
) -> list[npt.NDArray[np.float64]]:
    """Draw a random parameter set for `nn_controller`.

    THIS IS THE FUNCTION YOUR EA REPLACES. Instead of sampling weights from a
    normal distribution, your EA will search for them.

    Note the total parameter count printed by main(): that is the length of the
    flat vector an individual's genotype has to encode. Reshaping a flat
    genotype back into these matrices is on you.
    """
    return [
        RNG.normal(scale=0.5, size=(input_size, HIDDEN_SIZE)),
        RNG.normal(scale=0.5, size=(HIDDEN_SIZE, output_size)),
    ]


# ============================================================================ #
#  3. POSITION AND FITNESS
# ============================================================================ #
#
# The robot is spawned with a free joint, so data.qpos[0:3] IS the core's
# (x, y, z) world position. Read it before and after stepping - no tracker or
# bookkeeping needed. (`data.geom("robot1_core").xpos` works too.)
#
# ============================================================================ #


def get_core_position(data: mj.MjData) -> npt.NDArray[np.float64]:
    """Return the robot core's current (x, y, z) world position."""
    return np.asarray(data.qpos[0:3]).copy()


def fitness_function(
    initial_position: npt.NDArray[np.float64],
    final_position: npt.NDArray[np.float64],
) -> float:
    """Score one evaluation. LOWER IS BETTER.

    The plain version: how far is the robot from the target when time runs out?

    `initial_position` is unused here on purpose - it is passed in because the
    moment you want a less naive fitness you will need it. Some things worth
    thinking about (and, ideally, comparing in your report):
      * Distance *reduced* rather than distance remaining, so a robot that
        starts closer is not rewarded for standing still.
      * Penalising a robot that falls over or leaves the arena.
      * Whether the z-axis should count at all - a robot that jumps is not
        closer to the target in any way you care about.
    See `ariel.simulation.tasks.targeted_locomotion` for some worked variants.
    """
    target = np.asarray(TARGET_POSITION)
    return float(np.linalg.norm(final_position[:2] - target[:2]))


# ============================================================================ #
#  4. RUNNING ONE EVALUATION
# ============================================================================ #


def run_experiment(
    mode: ViewerTypes = MODE,
    genotype: list[float] | None = None,
    verbose: bool = True,
) -> float:
    """Set up the world, run one simulation, and return the fitness.

    This is the function your EA calls once per individual, with `mode` set
    to "simple" (headless).

    Returns
    -------
    float
        The fitness of this run. Lower is better.
    """
    # MuJoCo's control callback is a GLOBAL. Clear it. DO NOT REMOVE.
    mj.set_mjcb_control(None)

    # --- World and robot --------------------------------------------------- #
    world = build_world()
    robot = build_robot()

    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )

    # Compile the world into a model. USE AS IS.
    model = world.spec.compile()
    data = mj.MjData(model)

    # Put the simulation in a clean, known state before reading anything.
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    # --- Wire up the controller -------------------------------------------- #
    # Sizes are read from the compiled model, never hardcoded - they depend on
    # the body you chose in build_robot().
    input_size = len(controller_inputs(data))
    output_size = model.nu

    # ADDED: preserve the demo when no genotype is supplied.
    weights = (
        make_random_weights(input_size, output_size)
        if genotype is None
        else genotype_to_weights(genotype, input_size, output_size)
    )

    def control_callback(m: mj.MjModel, d: mj.MjData) -> None:
        """Compute and apply actions; MuJoCo calls this every physics step."""
        actions = nn_controller(m, d, weights)

        # DIRECT application (see the controller contract above).
        d.ctrl[:] = actions

        # DELTA application - comment out the line above and use these instead:
        # delta = 0.05
        # d.ctrl[:] += actions * delta
        # d.ctrl[:] = np.clip(d.ctrl, -np.pi / 2, np.pi / 2)

    # --- Record the starting point ----------------------------------------- #
    initial_position = get_core_position(data)

    # --- Run ---------------------------------------------------------------- #
    if mode != "no_control":
        mj.set_mjcb_control(control_callback)

    try:
        match mode:
            case "launcher":
                # Interactive window. Great for seeing what your robot does,
                # useless inside an evolutionary loop.
                viewer.launch(model=model, data=data)
            case "simple":
                # Headless. THIS is the one your EA uses.
                simple_runner(model, data, duration=SIM_DURATION)
            case "video":
                # Render to an mp4 - for the figures in your report.
                recorder = VideoRecorder(output_folder=str(DATA / "__videos__"))
                video_renderer(
                    model,
                    data,
                    duration=SIM_DURATION,
                    video_recorder=recorder,
                )
            case "frame":
                # A single image of the scene. Useful to check your spawn position
                # and that the robot is not clipping through the floor.
                single_frame_renderer(model, data, steps=1, show=True)
            case "no_control":
                # No controller attached: drag the hinges around by hand.
                viewer.launch(model=model, data=data)

    finally:
        mj.set_mjcb_control(None)

    # --- Score -------------------------------------------------------------- #
    final_position = get_core_position(data)
    fitness = fitness_function(initial_position, final_position)

    if verbose:
        console.log(f"start  : {np.round(initial_position, 3)}")
        console.log(f"end    : {np.round(final_position, 3)}")
        console.log(f"target : {np.round(TARGET_POSITION, 3)}")
        console.log(f"fitness: {fitness:.4f}   (lower is better)")

    return fitness if np.isfinite(fitness) else 1_000_000.0


def main() -> None:
    """Run a single demo evaluation with a randomly-weighted controller."""
    # A quick look at the size of the problem you are about to search.
    mj.set_mjcb_control(None)
    world = build_world()
    robot = build_robot()
    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )
    model = world.spec.compile()
    data = mj.MjData(model)

    input_size = len(controller_inputs(data))
    output_size = model.nu
    num_weights = (
        input_size * HIDDEN_SIZE
        + HIDDEN_SIZE * output_size
    )
    console.log(f"controller inputs (len(data.qpos)) : {input_size}")
    console.log(f"controller outputs (model.nu)      : {output_size}")
    console.log(f"genotype length (total weights)    : {num_weights}")

    run_experiment(MODE)


# ADDED: evolutionary experiment below; main() above remains as a demo.


# ============================================================================ #
#  YOUR JOB
# ============================================================================ #
#
# Everything above runs one robot with random weights. It will score badly, and
# it will score badly in a slightly different way every time you change SEED.
# Your task is to replace "random" with "evolved".
#
# Build a proper EA on top of `ariel.ec`. You are expected to use that module -
# it gives you the population/individual data model, the operators, and free
# persistence of every generation to a SQLite database, which you will want
# when it is time to plot convergence curves for the report.
#
#     from ariel.ec import EA, EAOperation, Individual, Population
#
# For a complete, runnable example of how those pieces fit together (a one-max
# EA with parent selection, crossover, mutation and survivor selection written
# as separate steps), read:
#
#     examples/new_EC_engine_example.py
#
# and the API documentation at:
#
#     https://ci-group.github.io/ariel/
#
# ---- EXPERIMENTAL RIGOUR ---------------------------------------------------
#
#   One run proves nothing. Repeat every configuration over several
#     independent seeds and report mean and spread.
#   Log best/mean/worst fitness per generation. The database `ariel.ec`
#     writes makes this straightforward.
#   Compare against a baseline. Random search with the same evaluation
#     budget is a simple, but reasonable choice; and it is nearly free to run.
#   Keep body, world, SIM_DURATION and fitness function identical across
#     everything you compare. Change one thing at a time.
#
# ============================================================================ #


def genotype_to_weights(
    genotype: list[float], input_size: int, output_size: int,
) -> list[npt.NDArray[np.float64]]:
    """Convert one flat list of real-valued genes into the original w1, w2."""
    genes = np.asarray(genotype, dtype=np.float64)
    cut = input_size * HIDDEN_SIZE
    required = cut + HIDDEN_SIZE * output_size
    if genes.size != required:
        raise ValueError(f"Expected {required} weights; got {genes.size}")
    return [
        genes[:cut].reshape(input_size, HIDDEN_SIZE),
        genes[cut:].reshape(HIDDEN_SIZE, output_size),
    ]


def make_individual(rng: np.random.Generator, length: int) -> Individual:
    individual = Individual()
    individual.genotype = rng.normal(0, 0.5, length).tolist()
    return individual


def get_genotype_length() -> int:
    mj.set_mjcb_control(None)
    world = build_world()
    robot = build_robot()
    world.spawn(robot.spec, position=SPAWN_POS, correct_collision_with_floor=True)
    model = world.spec.compile()
    data = mj.MjData(model)
    return len(controller_inputs(data)) * HIDDEN_SIZE + HIDDEN_SIZE * model.nu


def compare_run(question: int, variant: str, seed: int, quick: bool) -> dict:
    """One independent run; ariel.ec persists each generation in SQLite."""
    rng = np.random.default_rng(seed)
    set_seed(seed)
    mu = 4 if quick else MU
    lam = 8 if quick else LAMBDA
    min_generations = 2 if quick else MIN_GENERATIONS
    patience = 2 if quick else PATIENCE
    max_generations = 3 if quick else MAX_GENERATIONS
    length = get_genotype_length()
    output = DATA / f"q{question}" / variant / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)

    def evaluate(population: Population) -> Population:
        for ind in population.unevaluated:
            ind.fitness = run_experiment("simple", ind.genotype, verbose=EVAL_LOGS)
        return population

    def reproduce(population: Population) -> Population:
        parents = list(population.alive)
        for _ in range(lam):
            a = parents[int(rng.integers(len(parents)))]
            genes = np.asarray(a.genotype, dtype=float).copy()
            if question == 3 and variant == "recombination":
                b = parents[int(rng.integers(len(parents)))]
                mask = rng.random(length) < 0.5
                genes[mask] = np.asarray(b.genotype, dtype=float)[mask]
            # Both arms use exactly the same Gaussian mutation.
            genes += rng.normal(0.0, SIGMA, size=length)
            child = Individual()
            child.genotype = genes.tolist()
            population.append(child)
        return population

    def survivors(population: Population) -> Population:
        parents = [ind for ind in population.alive if ind.time_of_birth < generation]
        children = [ind for ind in population.alive if ind.time_of_birth == -1]
        # Offspring have not yet been committed, so their birth time is -1.
        candidates = children if question == 2 and variant == "comma" else parents + children
        chosen = sorted(candidates, key=lambda ind: ind.fitness)[:mu]
        keep = {id(ind) for ind in chosen}
        for ind in population.alive:
            if id(ind) not in keep:
                ind.alive = False
        return population

    initial = Population([make_individual(rng, length) for _ in range(mu)])
    evaluate(initial)
    db_path = output / "generations.db"
    ea = EA(initial, [EAOperation(reproduce), EAOperation(evaluate),
                      EAOperation(survivors)], db_file_path=db_path,
            db_handling="delete", is_maximisation=False, quiet=True)

    def current_stats(generation_number: int, incumbent: float) -> dict:
        scores = np.array([ind.fitness for ind in ea.population.alive], dtype=float)
        return {"generation": generation_number, "best": float(scores.min()),
                "mean": float(scores.mean()), "worst": float(scores.max()),
                "best_so_far": min(incumbent, float(scores.min()))}

    history = [current_stats(0, float("inf"))]
    incumbent = history[0]["best_so_far"]
    stagnant = 0
    stop_reason = "maximum generations reached"
    generation = 0
    for generation in range(1, max_generations + 1):
        ea.step()
        stats = current_stats(generation, incumbent)
        history.append(stats)
        if stats["best_so_far"] < incumbent - IMPROVEMENT:
            stagnant = 0
        else:
            stagnant += 1
        incumbent = min(incumbent, stats["best_so_far"])
        if generation >= min_generations and stagnant >= patience:
            stop_reason = "fitness plateau"
            break

    budget = mu + lam * generation
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    best = min(ea._fetch(only_alive=False), key=lambda ind: ind.fitness)
    (output / "best_genotype.json").write_text(
        json.dumps({"genotype": best.genotype, "fitness": best.fitness}),
        encoding="utf-8",
    )
    # Random search gets exactly the same number of simulator evaluations.
    baseline_rng = np.random.default_rng(seed + 1_000_000)
    baseline = float("inf")
    random_history = []
    for generation_number in range(generation + 1):
        count = mu if generation_number == 0 else lam
        for _ in range(count):
            candidate = make_individual(baseline_rng, length)
            baseline = min(baseline, run_experiment("simple", candidate.genotype,
                                                    verbose=False))
        random_history.append(baseline)
    with (output / "baseline.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best_so_far"])
        writer.writerows(enumerate(random_history))

    summary = {"question": question, "variant": variant, "seed": seed,
               "budget": budget, "generations": generation,
               "stop_reason": stop_reason, "best": incumbent,
               "baseline_best": baseline, "history": history,
               "random_history": random_history}
    (output / "summary.json").write_text(json.dumps(summary, indent=2),
                                           encoding="utf-8")
    return summary


def save_comparison(question: int, runs: list[dict]) -> None:
    """Average and SD at an evaluation budget shared by all runs."""
    output = DATA / f"q{question}"
    variants = ("plus", "comma") if question == 2 else ("mutation", "recombination")
    common_len = min(len(run["history"]) for run in runs)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "generation", "mean_best", "std_best",
                         "mean_random", "std_random", "number_of_runs"])
        for variant in variants:
            subset = [run for run in runs if run["variant"] == variant]
            x = np.arange(common_len)
            values = np.array([[r["history"][i]["best_so_far"]
                                for i in x] for r in subset])
            random = np.array([[r["random_history"][i]
                                for i in x] for r in subset])
            mean, sd = values.mean(axis=0), values.std(axis=0)
            rmean, rsd = random.mean(axis=0), random.std(axis=0)
            for i in x:
                writer.writerow([variant, i, mean[i], sd[i], rmean[i], rsd[i],
                                 len(subset)])
            plt.plot(x, mean, label=variant)
            plt.fill_between(x, mean - sd, mean + sd, alpha=0.16)
            plt.plot(x, rmean, linestyle="--", alpha=0.6,
                     label=f"random baseline ({variant})")
    plt.xlabel("Generations (equal evaluation budget across variants)")
    plt.ylabel("Best-so-far distance to target (lower is better)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "convergence.png", dpi=180)
    plt.close()


def experiment_main() -> None:
    global SIM_DURATION
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=int, choices=(2, 3))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--quick", action="store_true",
                        help="Pipeline test: tiny population, 3 generations, one seed")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.demo:
        main()
        return
    if args.question is None:
        parser.error("Choose --question 2 or --question 3 (or use --demo)")
    if args.quick:
        SIM_DURATION = 2.0  # debugging only; final experiments use 15 s.
    seeds = args.seeds[:1] if args.quick else args.seeds
    variants = ("plus", "comma") if args.question == 2 else (
        "mutation", "recombination")
    runs = []
    for variant in variants:
        for seed in seeds:
            result = compare_run(args.question, variant, seed, args.quick)
            runs.append(result)
            print(variant, seed, "best", round(result["best"], 4),
                  "random", round(result["baseline_best"], 4),
                  "evaluations", result["budget"], result["stop_reason"])
    save_comparison(args.question, runs)
    print("Saved output to", DATA / f"q{args.question}")


if __name__ == "__main__":
    experiment_main()
