import numpy as np
from skimage.io import imread
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import convolve2d
from PIL import Image
from esda.moran import Moran
from libpysal.weights import lat2W
from napari import Viewer, run
import pandas as pd
from scipy.stats import ttest_ind, normaltest, gaussian_kde
import os
from stacked_bar import get_symbol
plt.style.use("ggplot")
# get the default blue and red from the ggplot colorcycle
RED = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
BLUE = plt.rcParams["axes.prop_cycle"].by_key()["color"][1]
# transparent, red cmap
transRed = LinearSegmentedColormap.from_list("transRed", [(0, (0, 0, 0, 0)), (1, (1, 0, 0, 1))], N=2)


def aggregation_coefficient(data, n_subvolumes=(10, 10), voxel_size=(1, 1), threshold=False):
    """Calculates the aggregation coefficient for a 2D image."""
    subvolumes = get_subvolumes(data, n_subvolumes)
    N = np.prod(n_subvolumes)
    densities = np.zeros(N)
    thresh = threshold_otsu(data)
    for i, sv in enumerate(subvolumes):
        n, m = sv.shape
        if threshold:
            mask = sv >= thresh
        else:
            mask = sv
        biovolume = np.trapz(np.trapz(mask, dx=voxel_size[0], axis=0), dx=voxel_size[1])
        volume = np.prod(voxel_size) * (n - 1) * (m - 1)
        densities[i] = biovolume / volume
    densities = np.sort(densities)
    accumulation = np.cumsum(densities)
    S = np.sum(accumulation)
    T = np.sum(densities)
    S_max = T * (N + 1) / 2
    S_min = T
    A = (S_max - S) / (S_max - S_min)
    return A, data >= thresh


def calculate_morans_I(data):
    N, M = data.shape
    w = lat2W(N, M)  # Rook adjacency by default
    w.transform = "r"
    data_flat = data.flatten()
    moran = Moran(data_flat, w)
    return moran.I, moran.p_sim


def get_subvolumes(data, n_subvolumes):
    """Split a 2D image into subvolumes."""
    w = data.shape[0] // n_subvolumes[0]
    h = data.shape[1] // n_subvolumes[1]
    for i in range(n_subvolumes[0]):
        for j in range(n_subvolumes[1]):
            x = i * w
            y = j * h
            yield data[x : x + w, y : y + h]  # noqa E203
    return


def factors(n):
    factors = []
    for i in range(1, n + 1):
        if n % i == 0:
            factors.append(i)
    return factors


def ac_sensitivity(data):
    n_t = data.shape[0]
    n_subvolumes = [(i, i) for i in factors(data.shape[1])[1:]]
    # n_subvolumes = [(i, i) for i in range(2, 200, 4)]
    ac_data = np.zeros((n_t, len(n_subvolumes)))
    for j in range(n_t):
        for i, n in enumerate(n_subvolumes):
            ac_data[j, i] = aggregation_coefficient(data[j], n)
    return ac_data


def moran_sensitivity(data):
    n_t = data.shape[0]
    n_x, n_y = data.shape[1:]
    n_subvolumes = factors(data.shape[1])[:-1]
    # n_subvolumes = [(i, i) for i in range(2, 200, 4)]
    moran_data = np.zeros((n_t, len(n_subvolumes)))
    for j in range(n_t):
        for i, n in enumerate(n_subvolumes):
            moran_data[j, i] = calculate_morans_I(np.array(Image.fromarray(data[j]).resize((n_x // n, n_y // n))))[0]
    return n_subvolumes, moran_data


def smooth(data, k):
    kernel = np.ones((5, 5)) / 25
    for _ in range(k):
        data = convolve2d(data, kernel, mode="same")
    return data


def window_size_investigation(load_data=True):
    n = 360
    cell_data = imread("data/A1_6_Phase_1.ome.tiff")
    cell_data = np.array(Image.fromarray(cell_data[100]).resize((n, n)))
    X, Y = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    # approx delta by gaussian
    high_agg = np.exp((-(X**2 + Y**2) / 0.01))
    low_agg = np.ones((n, n))
    oscillating = np.sin(10 * np.pi * X) * np.sin(10 * np.pi * Y)
    noise = np.random.rand(n, n)
    smoothed = smooth(noise, 5)
    data = {
        "Gaussian": high_agg,
        "Uniform": low_agg,
        "Cosine": oscillating,
        "Noise": noise,
        "Smoothed": smoothed,
        "Cells": cell_data,
    }
    fig, ax = plt.subplots(2, 3)
    fig.set_size_inches(4, 3)
    ax = ax.flatten()
    for i, (k, v) in enumerate(data.items()):
        ax[i].imshow(v, cmap="gray", clim=[0, None])
        ax[i].set_title(k)
        ax[i].axis("off")
        ax[i].set_aspect("equal")
    plt.tight_layout()
    plt.savefig("media/clustering_test_bed.pdf")
    plt.show()
    if not load_data or not os.path.exists("data/ac_data.npy"):
        ac_data = {}
        moran_data = {}
        for k, v in data.items():
            print(f"Starting {k} aggregation coefficient")
            ac = ac_sensitivity(np.array([v]))
            print(f"Starting {k} moran")
            facts, moran = moran_sensitivity(np.array([v]))
            moran_data[k] = moran
            ac_data[k] = ac
        ac_data["n_sub"] = facts
        moran_data["n_sub"] = facts
        np.save("data/ac_data.npy", ac_data)
        np.save("data/moran_data.npy", moran_data)
    else:
        ac_data = np.load("data/ac_data.npy", allow_pickle=True).item()
        moran_data = np.load("data/moran_data.npy", allow_pickle=True).item()

    fig, ax = plt.subplots(2, 1)
    for k in data.keys():
        ax[0].plot(ac_data["n_sub"], ac_data[k].flatten(), label=k, marker="o", markersize=3)
        ax[1].plot(moran_data["n_sub"], moran_data[k].flatten(), label=k, marker="o", markersize=3)

    ax[0].legend(bbox_to_anchor=(1.25, 1.0))
    ax[0].set(xlim=(0, 125), title="Aggregation Coefficient")
    ax[1].legend(bbox_to_anchor=(1.25, 1.0))
    ax[1].set(title="Moran's I", xlabel="Stride in X")
    plt.tight_layout()
    plt.savefig("media/metrics.pdf")
    plt.show()


def ben_data():
    data = imread("data/A1_6_Phase_1.ome.tiff")
    nt, nx, ny = data.shape
    ac = [aggregation_coefficient(data[i], n_subvolumes=(9, 9)) for i in range(nt)]
    fig, ax = plt.subplots()
    ax.plot(ac)
    ax.set(xlabel="Time", ylabel="Aggregation Coefficient")
    ax.set_title("Aggregation Coefficient over Time")
    ax.set_ylim(0, 1)
    plt.show()


def parse_name(name):
    if name.endswith(".ome.tiff"):
        name = name[:-9]
        if "Phase" not in name:
            raise Warning("Not in livecyte format")
            return None
        well_no, roi_no, _, sec_no = name.split("_")

    else:
        raise Warning("Not an ome.tiff file")
        return None

    return well_no, roi_no, sec_no


def get_cell_type(well_no, well_dict={"WT": ["A1", "A2", "A3"], "R67K": ["C1", "C2", "C3"]}):
    for k, v in well_dict.items():
        if well_no in v:
            return k
    return None


def get_acs(folder, plot=False):
    ac_data = {}
    for f, data in load_images(folder):
        well_no, roi_no, sec_no = parse_name(f)
        print(f"Well: {well_no}, ROI: {roi_no}, Section: {sec_no}")
        cell_type = get_cell_type(well_no)
        print(f"Cell Type: {cell_type}")
        # find the second phase if it exists
        nt, nx, ny = data.shape
        print(f"Data shape: {data.shape}")
        ac_results = [aggregation_coefficient(data[i], n_subvolumes=(9, 9)) for i in range(nt)]
        confluencies = [np.sum(m) / (nx * ny) for a, m in ac_results]
        ac = [a[0] for a in ac_results]
        masks = np.array([a[1] for a in ac_results])
        ac_data[f] = [cell_type, ac, confluencies]
        if plot:
            viewer = Viewer()
            viewer.add_image(data)
            viewer.add_labels(masks)
            fig, ax = plt.subplots()
            ax.plot(ac)
            ax.set(xlabel="Time", ylabel="Aggregation Coefficient")
            ax.set_title(f"AC vs t for {cell_type}|{well_no}_{roi_no}_{sec_no}")
            ax.set_ylim(0, 1)
            plt.show()
    return ac_data


def load_images(folder):
    """Generator to lazily load and combine images."""
    fnames = [f for f in os.listdir(folder) if f.endswith(".ome.tiff")]
    well_nos, roi_nos, sec_nos = zip(*[parse_name(f) for f in fnames])
    u_fnames = set([(w, r) for w, r in zip(well_nos, roi_nos)])
    for w, r in u_fnames:
        m_fnames = sorted([f for f in fnames if f.startswith(f"{w}_{r}")])
        data = np.concatenate([imread(f"{folder}/{f}") for f in m_fnames], axis=0)
        f_name = f"{w}_{r}_Phase_Combined.ome.tiff"
        yield f_name, data


def box_violin_plot(wt_as, r67k_as):
    fig, ax = plt.subplots()
    fig.set_size_inches(4, 3)
    ax.boxplot([wt_as, r67k_as], labels=["WT", "R67K"], positions=[0, 1])
    ax.violinplot([wt_as, r67k_as], positions=[0, 1], showmedians=False, showextrema=False)
    # check normality
    z_wt, p_wt = normaltest(wt_as, nan_policy="omit")
    z_r67k, p_r67k = normaltest(r67k_as, nan_policy="omit")

    # calculate the p-value using a t-test
    t, p_ttest = ttest_ind(wt_as, r67k_as)
    ax.set_title("Mean Aggregation Coefficient")
    bottom, top = ax.get_ylim()
    y_range = top - bottom
    bar_height = (y_range * 0.05) + top
    bar_tips = bar_height - (y_range * 0.02)
    text_height = bar_height + (y_range * 0.02)
    ax.plot([0, 0, 1, 1], [bar_tips, bar_height, bar_height, bar_tips], lw=1, c="k")
    ax.text(0.5, text_height, get_symbol(p_ttest), ha="center", va="center")
    plt.tight_layout()
    plt.savefig("media/ac_boxplot.pdf")
    plt.show()
    stats_log = {
        "Number of WT samples": len(wt_as),
        "Number of R67K samples": len(r67k_as),
        "p-value": p_ttest,
        "Normality test WT": p_wt,
        "Normality test R67K": p_r67k,
    }
    print(f"{k}: {v} \n" for k, v in stats_log.items())
    pd.DataFrame(stats_log, index=[0]).to_csv("media/ac_stats.csv", index=False)


def get_array(ac_data):
    for ct in ["WT", "R67K"]:
        a_s, cons = zip(*[(a, con) for (c, a, con) in ac_data.values() if c == ct])
        max_len = max([len(a) for a in a_s])
        a_s_padded = np.array([np.pad(a, max_len - len(a), mode="constant", constant_values=(np.nan,)) for a in a_s])
        cons_padded = np.array([np.pad(c, max_len - len(c), mode="constant", constant_values=(np.nan,)) for c in cons])
        yield a_s_padded, cons_padded


def time_plot(ac_data):
    fig, ax = plt.subplots()
    fig.set_size_inches(4, 3)
    (wt_as, _), (r67k_as, _) = get_array(ac_data)
    t = np.arange(wt_as.shape[1])
    for a_s, name, color in zip([wt_as, r67k_as], ["WT", "R67K"], [BLUE, RED]):
        mu, sigma = np.nanmean(a_s, axis=0), np.nanstd(a_s, axis=0)
        ax.plot(t, mu, label=name, color=color)
        ax.fill_between(t, mu - sigma, mu + sigma, alpha=0.3, color=color)
    ax.set_title("Aggregation Coefficient over Time")
    ax.set(xlabel="Time", ylabel="Aggregation Coefficient")
    ax.legend()
    plt.tight_layout()
    plt.savefig("media/ac_time_plot.pdf")
    plt.show()


def aggregation_confluency_scatter_plot(ac_data):
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    fig.set_size_inches(6, 3)
    (wt_as, wt_cons), (r67k_as, r67k_cons) = get_array(ac_data)
    for i, (a_s, cons, name, cmap) in enumerate(zip([wt_as, r67k_as], [wt_cons, r67k_cons], ["WT", "R67K"], ["Blues", "Reds"])):
        xy = np.vstack([cons.flatten(), a_s.flatten()])
        z = gaussian_kde(xy)(xy)
        ax[i].scatter(cons.flatten(), a_s.flatten(), c=z, s=10, cmap=cmap)
        ax[i].set(xlabel="Confluency", ylabel="Aggregation Coefficient")
        ax[i].set_title(name)
    fig.suptitle = "Aggregation Coefficient vs Confluency"
    # ax.legend()
    plt.tight_layout()
    plt.savefig("media/ac_confluency_plot.pdf")
    plt.show()


def aggregation_confluency_shaded_error_plot(ac_data):
    fig, ax = plt.subplots()
    fig.set_size_inches(3, 3)
    (wt_as, wt_cons), (r67k_as, r67k_cons) = get_array(ac_data)
    na, nc = 100, 100
    for i, (a_s, cons, name, cmap) in enumerate(zip([wt_as, r67k_as], [wt_cons, r67k_cons], ["WT", "R67K"], ["Blues", "Reds"])):
        xy = np.vstack([cons.flatten(), a_s.flatten()])
        density = gaussian_kde(xy)
        a, c = np.linspace(0, 1, na), np.linspace(0, 1, nc)
        A, C = np.meshgrid(a, c)
        Z = density(np.vstack([C.flatten(), A.flatten()]))
        Z = Z.reshape((na, nc))
        # normalize Z by confluency
        Z /= np.trapz(y=Z, x=c, axis=0)
        ax.pcolor(C, A, Z, cmap=cmap)
        # ax.contourf(C, A, Z, cmap=cmap)
    fig.suptitle = "Aggregation Coefficient vs Confluency"
    # ax.legend()
    plt.tight_layout()
    plt.savefig("media/ac_confluency_shaded_error_plot.pdf")
    plt.show()


def show_videos(folder):
    for f, data in load_images(folder):
        viewer = Viewer()
        viewer.add_image(data)
        # viewer.window.add_dock_widget("text", name="Title")
        viewer.window.title = f
        run()


if __name__ == "__main__":
    # ac_data = get_acs("data/Images", plot=False)
    # np.save("data/ac_data.npy", ac_data)
    ac_data = np.load("data/ac_data.npy", allow_pickle=True).item()
    wt_as = np.array([np.mean(a) for (ct, a, _) in ac_data.values() if ct == "WT"])
    r67k_as = np.array([np.mean(a) for (ct, a, _) in ac_data.values() if ct == "R67K"])
    box_violin_plot(wt_as, r67k_as)
    time_plot(ac_data)
    aggregation_confluency_scatter_plot(ac_data)
    # aggregation_confluency_shaded_error_plot(ac_data)
