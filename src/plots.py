import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from metrics import energy_drift, momentum_drift, max_energy_error


# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results" / "data"
FIGS = ROOT / "results" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


# Style
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 120
})

COLORS = {
    "euler": "tab:red",
    "rk4": "tab:blue",
    "verlet": "tab:green"
}


# Data loading
def load_data(DATA=DATA):
    files = list(DATA.glob("*.csv"))
    data = {}

    for f in files:
        name = f.stem

        if "_" not in name or "runtime" in name:
            continue

        try:
            method, dt = name.split("_")
        except ValueError:
            continue

        df = pd.read_csv(f)

        # Remove first row (initial condition spike)
        df = df.iloc[1:].reset_index(drop=True)

        if method not in data:
            data[method] = []

        data[method].append((float(dt), df))

    for method in data:
        data[method].sort(key=lambda x: x[0])

    return data


# Save
def finalize_plot(filename, FIGS, show=True, save=False):
    path = FIGS / filename
    plt.tight_layout()
    
    if save:
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")

    if show:
        plt.show()
    else:
        plt.close()

# Energy drift by dt
def plot_energy_by_dt(data, FIGS, show=True, save=False):
    selected_dts = [1e-5, 1e-4, 1e-3]
    methods = list(data.keys())

    fig, axes = plt.subplots(1, len(selected_dts), figsize=(5 * len(selected_dts), 4), sharey=True)

    for ax, dt in zip(axes, selected_dts):
        for method in methods:
            for run_dt, df in data[method]:
                if np.isclose(run_dt, dt):
                    drift = energy_drift(df)
                    ax.plot(
                        df["t"],
                        drift,
                        label=method,
                        color=COLORS[method]
                    )

        ax.set_title(f"dt = {dt:.0e}")
        ax.set_xlabel("Time")
        ax.set_yscale("symlog")
        ax.grid(True)

    axes[0].set_ylabel("Relative Energy Drift")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    finalize_plot("energy_by_dt.png", FIGS, show, save)


# Final energy drift scaling
def plot_energy_scaling(data,FIGS, show=True, save=False):
    plt.figure()

    for method, runs in data.items():
        dts = []
        errors = []

        for dt, df in runs:
            if dt >= 1e-2:  # exclude breakdown regime
                continue

            err = max_energy_error(df)

            dts.append(dt)
            errors.append(err)

        plt.plot(
            dts,
            errors,
            marker="o",
            label=method,
            color=COLORS[method]
        )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Timestep (dt)")
    plt.ylabel("Max Relative Energy Error")
    plt.title("Energy Error Scaling (Stable Regime)")
    plt.legend()
    plt.grid(True)

    finalize_plot("energy_scaling.png", FIGS, show, save)


# Momentum drift by dt
def plot_momentum_by_dt(data, FIGS, show=True, save=False):
    selected_dts = [1e-5, 1e-4, 1e-3]
    methods = list(data.keys())

    fig, axes = plt.subplots(1, len(selected_dts), figsize=(5 * len(selected_dts), 4), sharey=True)

    for ax, dt in zip(axes, selected_dts):
        for method in methods:
            for run_dt, df in data[method]:
                if np.isclose(run_dt, dt):
                    drift = momentum_drift(df)
                    ax.plot(
                        df["t"],
                        drift,
                        label=method,
                        color=COLORS[method]
                    )

        ax.set_title(f"dt = {dt:.0e}")
        ax.set_xlabel("Time")
        ax.set_yscale("log")
        ax.grid(True)

    axes[0].set_ylabel("Momentum Drift")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    finalize_plot("momentum_by_dt.png", FIGS, show, save)

def plot_breakdown(data, FIGS, show=True, save=False):
    dt_target = 1e-2
    methods = list(data.keys())

    plt.figure()

    for method in methods:
        for run_dt, df in data[method]:
            if np.isclose(run_dt, dt_target):
                drift = energy_drift(df)
                plt.plot(
                    df["t"],
                    drift,
                    label=method,
                    color=COLORS[method]
                )

    plt.xlabel("Time")
    plt.ylabel("Relative Energy Drift")
    plt.title("Integrator Breakdown at dt = 1e-2")
    plt.yscale("symlog")
    plt.legend()
    plt.grid(True)

    finalize_plot("breakdown_dt_1e-2.png", FIGS, show, save)

# Main
if __name__ == "__main__":
    data = load_data()

    plot_energy_by_dt(data, FIGS, show=False, save=True)
    plot_energy_scaling(data, FIGS, show=False, save=True)
    plot_momentum_by_dt(data, FIGS, show=False, save=True)
    plot_breakdown(data, FIGS, show=False, save=True)    