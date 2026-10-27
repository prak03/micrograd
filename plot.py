import math

import matplotlib.pyplot as plt

BLUE = "#2a78d6"      # the network's prediction
GRAY = "#6b6a65"      # the true rule and reference lines
SHADE = "#e9e8e3"     # the range the network was trained on
INK = "#2b2a28"


def plot_results(train_losses, predict, baseline=None, x2=1.0,
                 x1_range=(-2.0, 2.0), path="results.png", truth=None, title=None):
    """Two panels: the training-loss curve, and the network's prediction
    versus the true rule along x1.

    train_losses: list of average training loss, one per epoch.
    predict:      function taking 4 plain floats [x1, x2, x3, x4] and
                  returning the network's prediction as a plain float.
    baseline:     loss of the best constant output (optional reference line).
    x2:           fixed value of x2 for the slice (x3 and x4 are set to 0).
    x1_range:     range of x1 to draw. Going past [-1, 1] shows extrapolation.
    """
    fig, (ax_loss, ax_pred) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel 1: loss curve
    epochs = range(len(train_losses))
    ax_loss.plot(epochs, train_losses, color=BLUE, linewidth=2)
    if baseline is not None:
        ax_loss.axhline(baseline, color=GRAY, linewidth=1, linestyle="--")
        ax_loss.annotate("constant-output baseline", xy=(len(train_losses) - 1, baseline),
                         xytext=(0, 5), textcoords="offset points",
                         ha="right", va="bottom", fontsize=9, color=GRAY)
    ax_loss.set_title("Average training loss", loc="left", color=INK)
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylim(bottom=0)

    # Panel 2: prediction vs truth along x1
    lo, hi = x1_range
    steps = 200
    xs = [lo + (hi - lo) * i / steps for i in range(steps + 1)]
    rule = truth or (lambda x: math.sin(math.pi * x[0]) * x[1])
    truth = [rule([x1, x2, 0.0, 0.0]) for x1 in xs]
    preds = [predict([x1, x2, 0.0, 0.0]) for x1 in xs]

    ax_pred.axvspan(-1, 1, color=SHADE, zorder=0)
    ax_pred.text(0, 1.12, "trained range", ha="center", fontsize=9, color=GRAY)
    ax_pred.plot(xs, truth, color=GRAY, linewidth=2, linestyle="--", label="true rule")
    ax_pred.plot(xs, preds, color=BLUE, linewidth=2, label="network")
    ax_pred.set_title(f"Prediction along x1  (x2 = {x2}, x3 = x4 = 0)", loc="left", color=INK)
    ax_pred.set_xlabel("x1")
    ax_pred.set_ylim(-1.25, 1.25)
    ax_pred.legend(frameon=False, loc="lower left")

    for ax in (ax_loss, ax_pred):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e4df", linewidth=0.8)
        ax.set_axisbelow(True)

    if title:
        fig.suptitle(title, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"saved {path}")
    plt.show()