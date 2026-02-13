import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate(CSV_PATH, OUT_PATH):
    df = pd.read_csv(CSV_PATH)

    fig, ax = plt.subplots()
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.0, 3.0)

    scat = ax.scatter([], [])

    def update(i):
        pts = []
        for k in df.columns:
            if k.startswith("x_") and k["alive"] != False:
                pid = k.split("_")[1]
                x = df[k].iloc[i]
                y = df[f"y_{pid}"].iloc[i]
                pts.append([x,y])

        scat.set_offsets(pts)
        return scat

    ani = FuncAnimation(fig, update, frames=len(df), interval=20)

    ani.save(OUT_PATH, writer="pillow")