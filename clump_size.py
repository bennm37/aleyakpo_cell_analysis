import numpy as np
import matplotlib.pyplot as plt

# remove small objects
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops
plt.style.use("ggplot")


def get_clumps(surface, z_thresh, min_clump_size):
    candidates = surface > z_thresh
    clumps = remove_small_objects(candidates, min_clump_size)
    return clumps


def quantify_clumps(clumps):
    labels = label(clumps)
    props = regionprops(labels)
    return props


if __name__ == "__main__":
    nx = 1000
    x, y = np.linspace(-2, 2, nx), np.linspace(-2, 2, nx)
    X, Y = np.meshgrid(x, y)
    surface = np.cos(5 * X) + np.cos(4 * X + 3 * Y) + np.sin(5 * Y)
    z_thresh = 1.2
    min_clump_size = 0.01 * nx ** 2
    clumps = get_clumps(surface, z_thresh, min_clump_size)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    ax[0].grid(False)
    ax[0].imshow(surface, cmap="viridis")
    ax[0].set(title="Surface")
    ax[1].grid(False)
    ax[1].imshow(clumps, cmap="viridis")
    ax[1].set(title="Clumps")
    plt.savefig("media/clumps.pdf")
    plt.show()
    props = quantify_clumps(clumps)
    print(f"Found {len(props)} clumps")
    print("Area\tMajor\tMinor")
    for prop in props:
        print(f"{prop.area/nx**2}\t{prop.major_axis_length/nx}\t{prop.minor_axis_length/nx}")
