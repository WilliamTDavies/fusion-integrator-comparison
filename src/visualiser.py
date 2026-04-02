from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_comparison(skip=None):
    ROOT = Path(__file__).resolve().parent.parent
    DATA = ROOT / "results" / "data"
    OUT = ROOT / "results" / "animations"
    OUT.mkdir(parents=True, exist_ok=True)

    # Group files by dt
    files = list(DATA.glob("*.csv"))
    groups = {}

    for f in files:
        name = f.stem
        if "_" not in name or "runtime" in name:
            continue

        method, dt = name.split("_")

        if dt not in groups:
            groups[dt] = {}

        groups[dt][method] = f

    # Process each dt group
    for dt, methods in groups.items():
        if not all(m in methods for m in ["euler", "rk4", "verlet"]):
            print(f"\nSkipping dt={dt} (missing method)")
            continue

        print(f"\nAnimating comparison for dt={dt}")

        dfs = {m: pd.read_csv(methods[m]) for m in methods}

        # Auto skip
        if skip is None:
            target_dt_vis = 0.01  # Adjust for speed

            actual_dt = dfs["euler"]["t"].iloc[1] - dfs["euler"]["t"].iloc[0]
            skip_val = max(1, int(target_dt_vis / actual_dt))
        else:
            skip_val = skip

        # Subsample
        dfs = {m: df.iloc[::skip_val].reset_index(drop=True) for m, df in dfs.items()}

        # Find particle IDs
        particle_ids = [
            col.split("_")[1]
            for col in dfs["euler"].columns
            if col.startswith("x_")
        ]

        # Global axis limits
        all_x = []
        all_y = []

        for df in dfs.values():
            x_vals = df.filter(like="x_").to_numpy().flatten()
            y_vals = df.filter(like="y_").to_numpy().flatten()

            all_x.append(x_vals)
            all_y.append(y_vals)

        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)

        # Remove NaN/Inf
        all_x = all_x[np.isfinite(all_x)]
        all_y = all_y[np.isfinite(all_y)]

        if len(all_x) == 0 or len(all_y) == 0:
            print(f"Skipping dt={dt} due to invalid data")
            continue

        # Remove NaN/Inf values
        all_x = all_x[np.isfinite(all_x)]
        all_y = all_y[np.isfinite(all_y)]

        x_min, x_max = -3, 3
        y_min, y_max = -3, 3

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        methods_order = ["euler", "rk4", "verlet"]

        scatters = []

        for ax, method in zip(axes, methods_order):
            ax.set_title(method.upper())
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect('equal')
            scat = ax.scatter([], [])
            scatters.append(scat)

        def update(frame):
            artists = []

            for ax, method, scat in zip(axes, methods_order, scatters):
                df = dfs[method]

                pts = []

                for pid in particle_ids:
                    x_col = f"x_{pid}"
                    y_col = f"y_{pid}"
                    alive_col = f"alive_{pid}"

                    # Skip if particle doesn't exist at this timestep
                    if x_col not in df.columns or y_col not in df.columns:
                        continue

                    x = df[x_col].iloc[frame]
                    y = df[y_col].iloc[frame]

                    # Skip NaNs (not yet created or removed)
                    if not np.isfinite(x) or not np.isfinite(y):
                        continue

                    if alive_col in df.columns and df[alive_col].iloc[frame] == 0:
                        continue

                    pts.append([x, y])

                scat.set_offsets(pts if pts else [])
                artists.append(scat)

            return artists

        frames = min(len(df) for df in dfs.values())

        ani = FuncAnimation(fig, update, frames=frames, interval=20, blit=True)

        out_path = OUT / f"compare_{dt}.gif"
        ani.save(out_path, writer="pillow", fps=30)

        print(f"Animation compare_{dt}.gif completed")

    print("All comparison animations complete.")

# Generate the animations
if __name__ == "__main__":
    animate_comparison()