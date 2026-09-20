"""Assignment 1 - simple TreeGenome evolutionary algorithm.

The program compares two variants of the same EA:
    1. point mutation
    2. subtree replacement mutation

"""

import copy
import random
from pathlib import Path
from typing import Literal

import mujoco as mj
import networkx as nx
import numpy as np
import torch
from mujoco import viewer

from tree_edit_distance import (
    distances_to_targets,
    mean_plus_std_tree_edit_distance,
    tree_edit_distance,
)

from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders._blueprint import (
    load_graph_from_json,
)
from ariel.ec import EA, EAOperation, Individual, Population, config
from ariel.ec.genotypes.tree.operators import (
    crossover_subtree,
    mutate_replace_node,
    mutate_subtree_replacement,
    random_tree,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.video_recorder import VideoRecorder


#  ----------------------------------------------------------------------------- #
#  CONFIGURATION
#  ----------------------------------------------------------------------------- #

type ViewerTypes = Literal["launcher", "video", "frame", "none"]

SCRIPT_NAME = Path(__file__).stem
HERE = Path(__file__).parent
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

TARGET_DIR: Path = HERE / "target_bodies"  # the bodies we must approach
NUM_OF_MODULES = 25 #Changing from 20 to 25 to allow the EA to evolve more complex bodies that can match the target bodies.
GENOTYPE = "tree"


#Small population to test the code .


# POP_SIZE = 8
# GENERATIONS = 2
# SEEDS = [10]
# MODE = "none"

#The required values for the assignment are as follows:
POP_SIZE = 50
GENERATIONS = 100
SEEDS = [10, 20, 30, 40, 50]
MODE: ViewerTypes = "none"
SPAWN_POS = [0.0, 0.0, 0.1]

P_CROSSOVER = 0.9
P_MUTATION = 0.1

# These two values are changed before every run.
TARGETS: list[nx.DiGraph] = []
MUTATION_VARIANT = "point"


#  ----------------------------------------------------------------------------- #
#  TARGETS AND FITNESS
#  ----------------------------------------------------------------------------- #

def load_targets(target_dir: Path = TARGET_DIR) -> list[nx.DiGraph]:
    """Load every target body graph from a directory.

    Returns
    -------
    list of nx.DiGraph
        One graph per JSON file, sorted by filename.

    Raises
    ------
    FileNotFoundError
        If the directory holds no target JSON files.
    """
    paths = sorted(target_dir.glob("*.json"))
    if not paths:
        msg = f"no target bodies found in {target_dir}"
        raise FileNotFoundError(msg)
    return [load_graph_from_json(p) for p in paths]


def fitness_function(
    body: nx.DiGraph,
    targets: list[nx.DiGraph],
) -> float:
    """Score one body against the whole target set. LOWER IS BETTER.

    Some things worth thinking about:
      * The std term charges for unevenness - body that is mediocre against every target
        and one that is excellent on most but bad on one can still land close
        in fitness, but the latter is penalized a bit more.
      * Nothing here rewards small bodies. Does your EA bloat? Should a size
        penalty be part of fitness, or is that the encoding's job?
    """
    return mean_plus_std_tree_edit_distance(body, targets)


#  ----------------------------------------------------------------------------- #
#  VISUALISATION FROM THE TEMPLATE
#  ----------------------------------------------------------------------------- #

def show_body(
    body: nx.DiGraph,
    mode: ViewerTypes = MODE,
    file_name: str = "body",
) -> None:
    """Build a body graph in MuJoCo and look at it.

    There is no controller and no physics worth speaking of - this exists so
    you can SEE what your fitness function is actually rewarding. Do this
    early and often. A number going down is not evidence that the bodies look
    anything like the targets.
    """
    if mode == "none":
        return

    # MuJoCo's control callback is a GLOBAL. Clear it. DO NOT REMOVE.
    mj.set_mjcb_control(None)

    world = SimpleFlatWorld()
    robot = construct_mjspec_from_graph(body)
    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )

    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    match mode:
        case "launcher":
            # Interactive window. Drag the modules around; nothing drives them.
            viewer.launch(model=model, data=data)
        case "frame":
            # A still image - the cheapest way to eyeball a body.
            save_path = str(DATA / f"{file_name}.png")
            single_frame_renderer(model, data, save=True, save_path=save_path)
            console.log(f"saved {save_path}")
        case "video":
            # Mostly useful for showing a body slumping under gravity.
            recorder = VideoRecorder(output_folder=str(DATA / "__videos__"))
            video_renderer(model, data, duration=5.0, video_recorder=recorder)


#  ----------------------------------------------------------------------------- #
#  EVOLUTIONARY ALGORITHM
#  ----------------------------------------------------------------------------- #

def make_individual() -> Individual:
    """Create one random tree individual."""
    genome = random_tree(max_modules=NUM_OF_MODULES)

    individual = Individual()
    individual.genotype = genome.to_dict()  #transform the genome to a dictionary representation
    individual.tags = {"selected": False}

    return individual


def evaluate(population: Population) -> Population:
    """Evaluate individuals that do not have fitness yet."""
    for individual in population.unevaluated:
        genome = TreeGenome.from_dict(individual.genotype) #Takes the dictionary and it returns a TreeGenome object
        body = genome.to_networkx()

        individual.fitness = fitness_function(body, TARGETS)

    return population


def parent_selection(population: Population) -> Population:
    """Select parents with the pairwise tournament from the example."""
    shuffled = population.alive.shuffle() #It shuffles the population to select parents randomly with the property of alive individuals.

    for individual in shuffled:
        individual.tags = {"selected": False}

    for i in range(0, len(shuffled) - 1, 2):
        individual_a = shuffled[i]
        individual_b = shuffled[i + 1]

        if individual_a.fitness_ is not None and individual_b.fitness_ is not None:
            # Lower fitness is better.
            if individual_a.fitness_ <= individual_b.fitness_:
                individual_a.tags = {"selected": True}
            else:
                individual_b.tags = {"selected": True}

    return population


def crossover(population: Population) -> Population:
    """Apply ARIEL subtree crossover to selected parents."""
    parents = population.where(
        lambda individual: individual.alive
        and bool(individual.tags.get("selected", False)),
    ).shuffle()

    for j in range(0, len(parents) - 1, 2):
        parent_a = parents[j]
        parent_b = parents[j + 1]

        genome_a = TreeGenome.from_dict(copy.deepcopy(parent_a.genotype))
        genome_b = TreeGenome.from_dict(copy.deepcopy(parent_b.genotype))


        if random.random() < P_CROSSOVER: #if the probability of crossover is less than the defined probability, then crossover is applied to the parents.
            child_genome_a, child_genome_b = crossover_subtree(genome_a,genome_b)

        else: #In case there is no crossover, the children are copies of the parents.
            child_genome_a = genome_a
            child_genome_b = genome_b

        child_a = Individual()
        child_a.genotype = child_genome_a.to_dict()
        child_a.tags = {"mutate": True}

        child_b = Individual()
        child_b.genotype = child_genome_b.to_dict()
        child_b.tags = {"mutate": True}

        population.extend([child_a, child_b])

    return population


def mutate(population: Population) -> Population:
    """Mutate offspring using the selected experimental variant."""
    offspring = population.where(
        lambda individual: individual.alive
        and bool(individual.tags.get("mutate", False)),
    )

    for individual in offspring:
        genome = TreeGenome.from_dict(copy.deepcopy(individual.genotype))

        if random.random() < P_MUTATION: #if the probability of mutation is less than the defined probability, then mutation is applied to the offspring.

            if MUTATION_VARIANT == "point":
                mutate_replace_node(genome)
            elif MUTATION_VARIANT == "subtree":
                mutate_subtree_replacement(
                    genome,
                    max_modules=NUM_OF_MODULES,
                )
            else:
                raise ValueError(
                    f"Unknown mutation variant: {MUTATION_VARIANT}",
                )

        individual.genotype = genome.to_dict()
        individual.requires_eval = True
        individual.tags = {"mutate": False}

    return population


def survivor_selection(population: Population) -> Population:
    """Keep the POP_SIZE individuals with the lowest fitness."""
    sorted_population = population.alive.sort(sort="min", attribute="fitness_")

    survivors = sorted_population[:POP_SIZE].to_list()
    survivor_ids = {id(individual) for individual in survivors}

    for individual in population.alive:
        if id(individual) not in survivor_ids:
            individual.alive = False

    return population


# ============================================================================ #
#  ONE INDEPENDENT RUN
# ============================================================================ #

def random_offspring(population: Population) -> Population:
    """Generate completely new random individuals for the baseline."""
    offspring = [make_individual() for _ in range(POP_SIZE)]
    population.extend(offspring)
    return population

def run_experiment(targets: list[nx.DiGraph],variant: str,seed: int) -> Individual:

    """Run one EA variant with one independent seed."""
    global TARGETS, MUTATION_VARIANT

    TARGETS = targets
    MUTATION_VARIANT = variant

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    config.target_population_size = POP_SIZE

    population = Population([
        make_individual()
        for _ in range(POP_SIZE)
    ])

    population = evaluate(population)

    # Chack if random-search baseline or EA for the mutations
    if variant == "random":
        operations = [
            EAOperation(random_offspring),
            EAOperation(evaluate),
            EAOperation(survivor_selection),
        ]
    else:
        operations = [
            EAOperation(parent_selection),
            EAOperation(crossover),
            EAOperation(mutate),
            EAOperation(evaluate),
            EAOperation(survivor_selection),
        ]

    run_folder = DATA / variant / f"seed_{seed}"
    run_folder.mkdir(parents=True, exist_ok=True)

    ea = EA(
        population,
        operations,
        num_steps=GENERATIONS,
        is_maximisation=False,
        db_file_path=run_folder / "database.db",
        db_handling="delete",
    )

    ea.run()

    return ea.get_solution("best", only_alive=True)


# ============================================================================ #
#  ENTRY POINT
# ============================================================================ #

def main() -> None:
    """Evolve populations and compare the two mutation variants."""
    targets = load_targets()

    # Keep the information printed by the original template.
    console.log(f"encoding      : {GENOTYPE}")
    console.log(f"module budget : {NUM_OF_MODULES}")
    console.log(f"population    : {POP_SIZE}")
    console.log(f"generations   : {GENERATIONS}")
    console.log(f"independent runs: {len(SEEDS)}")
    console.log(f"targets       : {len(targets)} bodies from {TARGET_DIR.name}")
    console.log(
        "target sizes  : "
        + ", ".join(str(target.number_of_nodes()) for target in targets),
    )

    # Keep the target-spread calculation from the original template.
    spread = [
        tree_edit_distance(target_a, target_b)
        for idx, target_a in enumerate(targets)
        for target_b in targets[idx + 1 :]
    ]
    console.log(
        f"target spread : mean pairwise distance {np.mean(spread):.2f}",
    )

    # --- Evolutionary experiments ----------------------------------------- #
    for variant in ("point", "subtree", "random"):
        for seed in SEEDS:
            console.log("")
            console.log(f"variant       : {variant}")
            console.log(f"seed          : {seed}")

            best = run_experiment(targets=targets,variant=variant,seed=seed)

            best_genome = TreeGenome.from_dict(best.genotype)
            best_body = best_genome.to_networkx()
            run_folder = DATA / variant / f"seed_{seed}"

            # database.db already contains all individuals. This separate JSON
            # makes the best final structure easy to inspect and reuse.
            best_genome.save_json(str(run_folder / "best_genome.json"))

            # Print the same information that the original main printed for
            # one random body, but now for the best evolved individual.
            console.log(f"best body     : {best_body.number_of_nodes()} modules")
            console.log(
                "per-target    : "
                + ", ".join(f"{d:.1f}" for d in distances_to_targets(best_body, targets)),
            )
            console.log(
                f"fitness       : {best.fitness_:.4f}   (lower is better)",
            )

            show_body(best_body,mode=MODE,file_name=f"best_{variant}_seed_{seed}")


 # This is outside both loops
    console.log("")
    console.log(f"All results saved in {DATA}" )           


if __name__ == "__main__":
    main()
