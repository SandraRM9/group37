#NOT DONE YET!!!


#Doing a 2-sided Wilcoxon test

import json

from scipy.stats import wilcoxon


#import stuff json
JSON_FILE = "plot_results_b.json"


def load_results():
    with open(JSON_FILE, "r") as file:
        return json.load(file)

def wilcoxon_test():

    results = load_results()

    for strategy in results.keys("strategy"):
        
        for seed in results.keys("seed"):
            best_fitness = results.get("best_fitness")

    return(best_fitness)


result_wilcoxon = wilcoxon_test()




# #final best fitness point, subtree, random
#     point_final_fitness = results["point"]["fitness"][:,-1]
#     subtree_final_fitness = results["subtree"]["fitness"][:,-1]
#     random_final_fitness = results["random"]["fitness"][:,-1]

#     #mean and sd for point, subtree, random
#     print(
#         "Mean point:", np.mean(point_final_fitness),
#         "\nSD point:", np.std(point_final_fitness),
#         "\nMean subtree:", np.mean(subtree_final_fitness),
#         "\nSD subtree:", np.std(subtree_final_fitness),
#         "\nMean random:", np.mean(random_final_fitness),
#         "\nSD random:", np.std(random_final_fitness),
#     )

#     #Wilcoxon test one-sided
#     result_wilcoxon_point_sub = wilcoxon(point_final_fitness, subtree_final_fitness, alternative="less") #point vs subtree
#     result_wilcoxon_point_random = wilcoxon(point_final_fitness, random_final_fitness, alternative="less") #point vs subtree
#     result_wilcoxon_sub_random = wilcoxon(subtree_final_fitness, random_final_fitness, alternative="less") #point vs subtree

#     print(
#         "Wilcoxon point subtree:", result_wilcoxon_point_sub,
#         "\nWilcoxon point random:", result_wilcoxon_point_random,
#         "\nWilcoxon subtree random:", result_wilcoxon_sub_random
#     )







if __name__ == "__main__":
    plot_seed_trajectories_by_strategy()