import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mannwhitneyu, normaltest

plt.style.use("ggplot")
GREEN = "#5D9C59"
RED = "#DF2E38"
BOX_KWARGS = {
    "boxprops": dict(facecolor="gray"),
    "whiskerprops": dict(color="gray"),
    "medianprops": dict(color="black"),
    "capprops": dict(color="black"),
}


def stacked_bar(data, ax, sort_by="G1"):
    n = data.shape[0]
    data["total"] = data.sum(axis=1)
    data.sort_values(by=sort_by, inplace=True)
    times = {col: data[col].values for col in data.columns if col != "total"}
    left = np.zeros(n)
    height = 0.5
    colors = [RED, GREEN]
    for color, (phase, time) in zip(colors, times.items()):
        ax.barh(range(n), time, height, left=left, label=phase, color=color)
        left += time
    return data, ax


def compare_stacked_bar(wt, mut):
    fig, ax = plt.subplots(2, 1, sharex=True)
    wt, ax[0] = stacked_bar(wt, ax[0])
    mut, ax[1] = stacked_bar(mut, ax[1])
    ax[0].set(yticks=[], title="Cell Cycle Phase Durations", ylabel="Wild Type")
    ax[0].legend(loc="upper right")
    ax[1].set(xlabel="Time (h)", yticks=[], ylabel="Mutant")
    plt.show()


def box_plot(wt, mut):
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(3, 6))
    ax[0].boxplot(
        [wt["G1"], mut["G1"]],
        positions=[0, 1],
        widths=0.5,
        patch_artist=True,
        labels=["WT", "Mut"],
        **BOX_KWARGS,
    )
    ax[1].boxplot(
        [wt["S/G2"], mut["S/G2"]], positions=[0, 1], widths=0.5, patch_artist=True, labels=["WT", "Mut"], **BOX_KWARGS
    )
    ax[0].set(ylabel="G1 Phase (h)")
    ax[1].set(ylabel="S/G2 Phase (h)")
    plt.tight_layout()
    plt.savefig("media/boxplot.pdf")
    plt.show()


def violin_plot(wt, mut):
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(3, 6))
    ax[0].violinplot(
        [clean(wt["G1"]), clean(mut["G1"])],
        positions=[0, 1],
        widths=0.5,
        showmedians=False,
        showextrema=False,
    )
    ax[0].boxplot(
        [clean(wt["G1"]), clean(mut["G1"])],
        positions=[0, 1],
        widths=0.2,
        patch_artist=True,
        labels=["WT", "Mut"],
        **BOX_KWARGS,
    )
    ax[1].violinplot(
        [clean(wt["S/G2"]), clean(mut["S/G2"])],
        positions=[0, 1],
        widths=0.5,
        showmedians=False,
        showextrema=False,
    )
    ax[1].boxplot(
        [clean(wt["S/G2"]), clean(mut["S/G2"])],
        positions=[0, 1],
        widths=0.2,
        patch_artist=True,
        labels=["WT", "Mut"],
        **BOX_KWARGS,
    )
    ax[0].set(ylabel="G1 Phase (h)")
    ax[1].set(ylabel="S/G2 Phase (h)", xticks=[0, 1], xticklabels=["WT", "Mut"])
    # significance bars
    p_G1 = mannwhitneyu(clean(wt["G1"]), clean(mut["G1"]))[1]
    p_SG2 = mannwhitneyu(clean(wt["S/G2"]), clean(mut["S/G2"]))[1]
    for i, p in enumerate([p_G1, p_SG2]):
        bottom, top = ax[i].get_ylim()
        y_range = top - bottom
        bar_height = (y_range * 0.05) + top
        bar_tips = bar_height - (y_range * 0.02)
        ax[i].plot(
            [0, 0, 1, 1],
            [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k'
        )
        ax[i].text(0.5, 0.98, get_symbol(p), ha="center", va="center", transform=ax[i].transAxes)

    # external key explaining p values
    # key = {"*": "p < 0.05", "**": "p < 0.01", "***": "p < 0.001", "ns": "p > 0.05"}
    plt.tight_layout()
    plt.savefig("media/violinplot.pdf")
    plt.show()


def normality_test(wt, mut):
    data = {"WT_G1": wt["G1"], "WT_S/G2": wt["S/G2"], "Mut_G1": mut["G1"], "Mut_S/G2": mut["S/G2"]}
    for key, arr in data.items():
        print(f"Normality test for {key}")
        z, p = normaltest(arr, nan_policy="omit")
        print(f"z-score: {z}")
        print(f"p-value: {p}")
        if p < 0.05:
            print("Significantly non-normal distribution.")
        else:
            print("Not significantly non-normal.")


def mann_whitney_u_test(arr1, arr2):
    test_stat, p = mannwhitneyu(arr1, arr2)
    print(f"test statistic: {test_stat}")
    print(f"p-value: {p}")
    if p < 0.05:
        print("Significantly different distributions.")
    else:
        print("Not significantly different.")


def clean(arr):
    return arr[~np.isnan(arr)]


def get_symbol(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


if __name__ == "__main__":
    wt = pd.read_csv("data/wt.tsv", sep="\t")
    mut = pd.read_csv("data/mut.tsv", sep="\t")
    compare_stacked_bar(wt, mut)
    # box_plot(wt, mut)
    # normality_test(wt, mut)
    # violin_plot(wt, mut)
