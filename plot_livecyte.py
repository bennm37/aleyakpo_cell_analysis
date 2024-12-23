import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os


def animate(df):
    unique_frames = sorted(df["Frame"].unique())
    fig, ax = plt.subplots()
    df["Thickness (\u00b5m)"] = df["Volume (\u00b5m\u00b3)"] / df["Area (\u00b5m\u00b2)"]
    scatter = ax.scatter(
        [],
        [],
        s=[],
        c=[],
        cmap="viridis",
        edgecolor="black",
        vmin=df["Thickness (\u00b5m)"].min(),
        vmax=df["Thickness (\u00b5m)"].max(),
    )

    def init():
        ax.set_xlim(df["Position X (\u00b5m)"].min(), df["Position X (\u00b5m)"].max())
        ax.set_ylim(df["Position Y (\u00b5m)"].min(), df["Position Y (\u00b5m)"].max())
        c = plt.colorbar(scatter, ax=ax, label="Thickness \u00b5m")
        return (scatter, c)

    def update(frame):
        frame_data = df[df["Frame"] == frame]
        scatter.set_offsets(frame_data[["Position X (\u00b5m)", "Position Y (\u00b5m)"]])
        scatter.set_sizes(frame_data["Radius (\u00b5m)"] ** 2)  # Scale radius for visualization
        scatter.set_array(frame_data["Thickness (\u00b5m)"])
        return (scatter,)

    ani = FuncAnimation(fig, update, frames=unique_frames, init_func=init, interval=60, repeat=False)
    return ani


# Example Usage
if __name__ == "__main__":
    fname = "data/livecyte/2023_07_24 HEK cells FUCCI WT vs R67K_A1_8_Phase-FullFeatureTable.csv"
    name = os.path.basename(fname).split(".")[0]
    name = name.split("_Phase")[0]
    print(name)
    df = pd.read_csv("data/livecyte/2023_07_24 HEK cells FUCCI WT vs R67K_A1_8_Phase-FullFeatureTable.csv", header=1)
    ani = animate(df)
    ani.save(f"media/{name}.gif")
