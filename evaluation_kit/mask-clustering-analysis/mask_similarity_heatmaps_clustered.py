import numpy as np
import matplotlib
matplotlib.use("Agg")  # remove if you want an interactive window
import matplotlib.pyplot as plt

#path = "./log/similarity/ct14/similarity_cosine.npy"
path = "./log/similarity/ct14_0.5_freq5/similarity_cosine.npy"
data = np.load(path, allow_pickle=True).item()

task_order = [1, 3, 5, 7, 9, 11, 13, 2, 4, 6, 8, 10, 12, 14]
idx = [data["task_ids"].index(t) for t in task_order]

layer_names = data["layer_names"]
layer_sims = data["layer_sims"]

fig, axes = plt.subplots(1, len(layer_names), figsize=(24, 5), constrained_layout=True)
for ax, layer_name in zip(axes, layer_names):
    mat = layer_sims[layer_name][np.ix_(idx, idx)]
    im = ax.imshow(mat, cmap="Oranges", vmin=0, vmax=1)
    ax.set_title(layer_name)
    ax.set_xticks(range(len(task_order)))
    ax.set_yticks(range(len(task_order)))
    ax.set_xticklabels(task_order, rotation=90)
    ax.set_yticklabels(task_order)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)

# single shared colorbar
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.65, pad=0.02)
fig.suptitle("Mask cosine similarity across tasks clustered", fontsize=14)
plt.savefig('./log/similarity/ct14_0.5_freq5/similarity_cosine_cluster.pdf')  # or plt.savefig("mask_similarity_reordered.png", dpi=300)
