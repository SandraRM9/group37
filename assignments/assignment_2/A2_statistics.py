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
    mu_plus_lambda_generations = []
    mu_comma_lambda_generations = []
    random_search_generations = []
    mu_comma_lambda_stop = []
    mu_plus_lambda_stop = []
    random_search_stop = []

    results = load_results()
    
    for run in results:
        if run.get("strategy") == "mu_plus_lambda":
            mu_plus_lambda_fitness.append(run.get("best_fitness"))
            mu_plus_lambda_generations.append(run.get("generations_run"))
            mu_plus_lambda_stop.append(run.get("stopping_reason"))
        elif run.get("strategy") == "mu_comma_lambda":
            mu_comma_lambda_fitness.append(run.get("best_fitness"))
            mu_comma_lambda_generations.append(run.get("generations_run"))
            mu_comma_lambda_stop.append(run.get("stopping_reason"))
        elif run.get("strategy") == "random_search":
            random_search_fitness.append(run.get("best_fitness"))
            random_search_generations.append(run.get("generations_run"))
            random_search_stop.append(run.get("stopping_reason"))

    print(
        "\nMean mu_plus_lambda:", np.mean(mu_plus_lambda_fitness),
        "\nSD mu_plus_lambda:", np.std(mu_plus_lambda_fitness),
        "\nMean generations ran mu_plus_lambda:", np.mean(mu_plus_lambda_generations),
        "\nReason for stopping mu_plus_lambda: plateau:", mu_plus_lambda_stop.count('plateau'), ", target reached:", mu_plus_lambda_stop.count("target_reached"),
        "\nMean mu_comma_lambda:", np.mean(mu_comma_lambda_fitness),
        "\nSD mu_comma_lambda:", np.std(mu_comma_lambda_fitness),
        "\nMean generations ran mu_comma_lambda:", np.mean(mu_comma_lambda_generations),
        "\nReason for stopping mu_comma_lambda: plateau:",  mu_comma_lambda_stop.count('plateau'), ", target reached:", mu_comma_lambda_stop.count("target_reached"),
        "\nMean random:", np.mean(random_search_fitness),
        "\nSD random:", np.std(random_search_fitness),
        "\nMean generations ran random_search:", np.mean(random_search_generations), 
        "\nReason for stopping random_search: plateau:", random_search_stop.count("plateau"), ", target reached:", random_search_stop.count("target_reached"),
        "\n"
    )

    result_wilcoxon_plus_comma = wilcoxon(mu_plus_lambda_fitness, mu_comma_lambda_fitness, alternative = "two-sided")
    result_wilcoxon_plus_random = wilcoxon(mu_plus_lambda_fitness, random_search_fitness, alternative = "two-sided")
    result_wilcoxon_comma_random = wilcoxon(mu_comma_lambda_fitness, random_search_fitness, alternative = "two-sided")

    
    print(
        "Result wilcoxon plus_comma:", result_wilcoxon_plus_comma,
        "\nResult wilcoxon plus_random:", result_wilcoxon_plus_random,
        "\nResult wilcoxon comma_random", result_wilcoxon_comma_random,
        "\n"
    )

    return(result_wilcoxon_plus_comma, result_wilcoxon_plus_random, result_wilcoxon_comma_random)

    

result_wilcoxon = wilcoxon_test()





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