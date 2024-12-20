import numpy as np
from skimage.io import imread
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import convolve2d
from PIL import Image
from esda.moran import Moran
from libpysal.weights import lat2W
import os

plt.style.use("ggplot")
# transparent, red cmap
transRed = LinearSegmentedColormap.from_list("transRed", [(0, (0, 0, 0, 0)), (1, (1, 0, 0, 1))], N=2)


def aggregation_coefficient(data, n_subvolumes=(10, 10), voxel_size=(1, 1), threshold=True):
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
    return A


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


if __name__ == "__main__":
    ben_data()
"/Volumes/lab-vincentj/home/users/aleyakb/Cells/2022_10_28 NucView HEK T vs R67K Apoptosis/2022_10_28 NucView HEK T vs R67K Apoptosis - Outputs/Raw Data/2022-10-28_10-39-51/Images/A1_6_FITC_1.ome.tiff"
