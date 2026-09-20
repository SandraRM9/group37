import json
import sqlite3
from pathlib import Path

import pandas as pd

from ariel.ec.genotypes.tree.tree_genome import TreeGenome


database_path = (
    Path("__data__")
    / "A1_complete_tree_b"
    / "random"
    / "seed_10"
    / "database.db"
)


def count_modules(stored_genotype):
    genotype = json.loads(stored_genotype)
    genome = TreeGenome.from_dict(genotype)
    return genome.to_networkx().number_of_nodes()


with sqlite3.connect(database_path) as connection:
    data = pd.read_sql("SELECT * FROM individual", connection)

# Count modules for every individual stored in this run
data["body_size"] = data["genotype_"].apply(count_modules)

initial = data[data["time_of_birth"] == 0]

print("INITIAL RANDOM INDIVIDUALS:")
print(initial["body_size"].value_counts().sort_index())

print("\nALL INDIVIDUALS CREATED DURING THE RUN:")
print(data["body_size"].value_counts().sort_index())