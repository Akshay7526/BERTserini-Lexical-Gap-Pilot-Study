import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("results.json") as f:
    results = json.load(f)

groups = ["matched", "paraphrased"]
metrics = ["recall@1", "recall@3", "recall@5"]
methods = ["bm25", "lsa"]
method_labels = {"bm25": "BM25 (lexical)", "lsa": "TF-IDF+LSA (semantic proxy)"}
colors = {"bm25": "#F0A500", "lsa": "#1E2761"}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=300)

for ax, grp in zip(axes, groups):
    x = np.arange(len(metrics))
    width = 0.35
    for i, m in enumerate(methods):
        vals = [results[m][grp][metric] for metric in metrics]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=method_labels[m], color=colors[m])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["Recall@1", "Recall@3", "Recall@5"], fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title(f"{grp.capitalize()} questions (n=24)", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Score", fontsize=11)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06), fontsize=11, frameon=False)
fig.tight_layout()
fig.savefig("retrieval_results.png", dpi=300, bbox_inches="tight", facecolor="white")
print("saved")
