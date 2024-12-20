import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt


def run_logrank_test(df):
    group_0 = df[df["group"] == 0]
    group_1 = df[df["group"] == 1]
    kmf_0 = KaplanMeierFitter()
    kmf_1 = KaplanMeierFitter()
    kmf_0.fit(group_0["stop"], event_observed=group_0["event"], entry=group_0["start"], label="Group 0")
    kmf_1.fit(group_1["stop"], event_observed=group_1["event"], entry=group_1["start"], label="Group 1")
    results = logrank_test(
        group_0["stop"],
        group_1["stop"],
        event_observed_A=group_0["event"],
        event_observed_B=group_1["event"],
        entry_A=group_0["start"],
        entry_B=group_1["start"],
    )
    print(f"Log-rank test p-value: {results.p_value}")
    if results.p_value < 0.05:
        print("The survival rates of the two groups are significantly different.")
    else:
        print("No significant difference in survival rates between the groups.")
    print(f"Log-rank test p-value: {results.p_value}")
    if results.p_value < 0.05:
        print("The survival rates of the two groups are significantly different.")
    else:
        print("No significant difference in survival rates between the groups.")
    plot_km(kmf_0, kmf_1)


def plot_km(kmf_0, kmf_1):
    plt.figure(figsize=(8, 6))
    kmf_0.plot_survival_function(ci_show=True)
    kmf_1.plot_survival_function(ci_show=True)
    plt.title("Survival Curves")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.legend()
    plt.grid()
    plt.show()


def format_survival_data(n_cells, birth_death_times):
    """n_cells is an (nt,) array of the number of cells at each time point.
       birth_death_times is an (nt, 3) array of birth and death times for cells that die, 
       giving the start time, death time and a binary indicator (1 for birth, 0 for censored)
       for the start time."""
    start = []
    stop = []
    event = []
    nt = n_cells.shape[0]
    t = np.linspace(0, nt - 1, nt)
    for i in range(nt):
        n_born = n_cells[i]
        start.extend([i] * n_cells[i])
        stop.extend([nt-1] * n_cells[i])
        event.extend([1] * n_cells[i])


if __name__ == "__main__":
    data = {
        "start": np.array([0, 0, 2, 3, 5, 0, 1, 4, 6, 0]),
        "stop": np.array([5, 6, 8, 9, 12, 7, 10, 11, 13, 5]),
        "event": np.array([1, 1, 0, 1, 1, 0, 1, 0, 1, 1]),
        "group": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
    }
    df = pd.DataFrame(data)
    run_logrank_test(df)
