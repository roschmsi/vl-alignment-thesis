import matplotlib.pyplot as plt
from pathlib import Path

OURS_LABEL = "SOTAlign (Ours)"
BASELINE_LABEL = "SAIL"

OUTDIR = Path("plots")
OUTDIR.mkdir(parents=True, exist_ok=True)


def extract_series(results, metric_key, which="ours"):
    xs = sorted(results.keys())
    ys = [results[x][which][metric_key] for x in xs]
    return xs, ys


def nice_xtick_label(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:g}M"
    if n >= 1_000:
        return f"{n/1_000:g}k"
    return str(n)


def plot_num_anchors(results, metric_key, title, ylabel, filename=None, dpi=200):
    xs, ys_ours = extract_series(results, metric_key, which="ours")
    _, ys_base = extract_series(results, metric_key, which="baseline")

    fig, ax = plt.subplots()

    ax.plot(xs, ys_ours, marker="o", label=OURS_LABEL, color="tab:blue")
    if results["baseline"] is not None:
        ax.plot(
            xs,
            ys_base,
            marker="s",
            linestyle="--",
            label=BASELINE_LABEL,
            color="tab:orange",
        )

    # ax.set_title(title)
    ax.set_xlabel("Number of anchors")
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


if __name__ == "__main__":

    results_anchors = {
        1_000: {
            "ours": {"t2i_r1": 11.9, "i2t_r1": 14.0, "cls_acc": 18.2},
            "baseline": {"t2i_r1": 6.4, "i2t_r1": 8.1, "cls_acc": 13.7},
        },
        10_000: {
            "ours": {"t2i_r1": 22.0, "i2t_r1": 29.6, "cls_acc": 39.5},
            "baseline": {"t2i_r1": 18.9, "i2t_r1": 23.5, "cls_acc": 33.5},
        },
        100_000: {
            "ours": {"t2i_r1": 26.9, "i2t_r1": 35.4, "cls_acc": 47.5},
            "baseline": {"t2i_r1": 26.7, "i2t_r1": 35.2, "cls_acc": 46.6},
        },
        1_000_000: {
            "ours": {"t2i_r1": 27.9, "i2t_r1": 37.2, "cls_acc": 48.6},
            "baseline": {"t2i_r1": 28.5, "i2t_r1": 37.1, "cls_acc": 48.1},
        },
    }

    p1 = plot_num_anchors(
        results_anchors,
        "t2i_r1",
        "COCO T2I Retrieval",
        "R@1",
        filename="t2i_r1_num_anchors.png",
    )
    p2 = plot_num_anchors(
        results_anchors,
        "i2t_r1",
        "COCO I2T Retrieval",
        "R@1",
        filename="i2t_r1_num_anchors.png",
    )
    p3 = plot_num_anchors(
        results_anchors,
        "cls_acc",
        "ImageNet Classification",
        "Accuracy (%)",
        filename="cls_acc_num_anchors.png",
    )

    print("Saved:")
    print(p1)
    print(p2)
    print(p3)
