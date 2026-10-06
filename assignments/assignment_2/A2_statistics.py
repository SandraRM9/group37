#NOT DONE YET!!!


#Doing a 2-sided Wilcoxon test

import json
import numpy as np
from scipy.stats import wilcoxon


#import stuff json
JSON_FILE = "plot_results_b.json"


def load_results():
    with open(JSON_FILE, "r") as file:
        return json.load(file)

def wilcoxon_test():
    mu_plus_lambda_fitness = []
    mu_comma_lambda_fitness = []
    random_search_fitness = []
    results = load_results()
    
    for run in results:
        if run.get("strategy") == "mu_plus_lambda":
            mu_plus_lambda_fitness.append(run.get("best_fitness"))
        elif run.get("strategy") == "mu_comma_lambda":
            mu_comma_lambda_fitness.append(run.get("best_fitness"))
        elif run.get("strategy") == "random_search":
            random_search_fitness.append(run.get("best_fitness"))

    print(
        "Mean mu_plus_lambda:", np.mean(mu_plus_lambda_fitness),
        "\nSD mu_plus_lambda:", np.std(mu_plus_lambda_fitness),
        "\nMean mu_comma_lambda:", np.mean(mu_comma_lambda_fitness),
        "\nSD mu_comma_lambda:", np.std(mu_comma_lambda_fitness),
        "\nMean random:", np.mean(random_search_fitness),
        "\nSD random:", np.std(random_search_fitness),
    )

    result_wilcoxon_plus_comma = wilcoxon(mu_plus_lambda_fitness, mu_comma_lambda_fitness)
    result_wilcoxon_plus_random = wilcoxon(mu_plus_lambda_fitness, random_search_fitness)
    result_wilcoxon_comma_random = wilcoxon(mu_comma_lambda_fitness, random_search_fitness)

    print(
        "Result wilcoxon plus_comma:", result_wilcoxon_plus_comma,
        "\nResult wilcoxon plus_random:", result_wilcoxon_plus_random,
        "\nResult wilcoxon comma_random", result_wilcoxon_comma_random
    )

    return(result_wilcoxon_plus_comma, result_wilcoxon_plus_random, result_wilcoxon_comma_random)

    
        




result_wilcoxon = wilcoxon_test()

print(result_wilcoxon)




#     #Wilcoxon test one-sided
#     result_wilcoxon_point_sub = wilcoxon(point_final_fitness, subtree_final_fitness, alternative="less") #point vs subtree
#     result_wilcoxon_point_random = wilcoxon(point_final_fitness, random_final_fitness, alternative="less") #point vs subtree
#     result_wilcoxon_sub_random = wilcoxon(subtree_final_fitness, random_final_fitness, alternative="less") #point vs subtree

#     print(
#         "Wilcoxon point subtree:", result_wilcoxon_point_sub,
#         "\nWilcoxon point random:", result_wilcoxon_point_random,
#         "\nWilcoxon subtree random:", result_wilcoxon_sub_random
#     )







# if __name__ == "__main__":
#     plot_seed_trajectories_by_strategy()