import json
import matplotlib.pyplot as plt
import numpy as np

with open("plot_results.json", "r") as f:
    data = json.load(f)

div_plus = [r["final_diversity"] for r in data if r["strategy"] == "mu_plus_lambda"]
div_comma = [r["final_diversity"] for r in data if r["strategy"] == "mu_comma_lambda"]

plt.figure(figsize=(4.5, 4))
box = plt.boxplot([div_plus, div_comma], 
                  tick_labels=[r"$(\mu + \lambda)$", r"$(\mu, \lambda)$"],
                  patch_artist=True,
                  medianprops=dict(color="black", linewidth=1.5),
                  boxprops=dict(facecolor="#d0e1f9", edgecolor="black"))


box['boxes'][0].set_facecolor('#1f77b4')
box['boxes'][0].set_alpha(0.7)
box['boxes'][1].set_facecolor('#ff7f0e')
box['boxes'][1].set_alpha(0.7)

plt.ylabel("Final Diversity", fontsize=10)
plt.title("Population Diversity Comparison", fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("diversity_boxplot.png", dpi=300)
plt.show()