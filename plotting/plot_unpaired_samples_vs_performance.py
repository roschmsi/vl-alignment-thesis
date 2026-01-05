import matplotlib.pyplot as plt
from pathlib import Path

results = {
    10_000: {"t2i_r1": 21.1, "i2t_r1": 27.7, "cls_acc": 37.0},
    100_000: {"t2i_r1": 21.8, "i2t_r1": 29.4, "cls_acc": 37.9},
    1_000_000: {"t2i_r1": 22.0, "i2t_r1": 29.6, "cls_acc": 39.5},
    2_800_000: {"t2i_r1": 22.2, "i2t_r1": 29.6, "cls_acc": 39.3},
}

baseline = {
    "t2i_r1": 18.9,
    "i2t_r1": 23.5,
    "cls_acc": 33.5,
}

OURS_LABEL = "AnchorVLM (Ours)"
BASELINE_LABEL = "SAIL"

OUTDIR = Path("plots")
OUTDIR.mkdir(parents=True, exist_ok=True)


def extract_series(metric_key, which="ours"):
    xs = sorted(results_anchors.keys())
    ys = [results_anchors[x][which][metric_key] for x in xs]
    return xs, ys


def nice_xtick_label(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:g}M"
    if n >= 1_000:
        return f"{n/1_000:g}k"
    return str(n)


def plot_metric(metric_key, title, ylabel, filename=None, dpi=200):
    xs, ys = extract_series(metric_key)

    fig, ax = plt.subplots()
    ax.plot(xs, ys, marker="o", label=OURS_LABEL, color="tab:blue")

    b = baseline.get(metric_key, None)
    if b is not None:
        ax.axhline(
            b, linestyle="--", linewidth=1.5, label=BASELINE_LABEL, color="tab:orange"
        )

    ax.set_title(title)
    ax.set_xlabel("Number of unpaired samples")
    ax.set_ylabel(ylabel)

    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([nice_xtick_label(x) for x in xs])

    ax.legend()
    fig.tight_layout()

    outpath = OUTDIR / filename
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return outpath


p1 = plot_metric(
    "t2i_r1", "COCO T2I Retrieval", "R@1", filename="t2i_r1_num_unpaired.png"
)
p2 = plot_metric(
    "i2t_r1", "COCO I2T Retrieval", "R@1", filename="i2t_r1_num_unpaired.png"
)
p3 = plot_metric(
    "cls_acc",
    "ImageNet Classification",
    "Accuracy (%)",
    filename="cls_acc_num_unpaired.png",
)

print("Saved:")
print(p1)
print(p2)
print(p3)
