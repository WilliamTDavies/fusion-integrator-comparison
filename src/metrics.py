import pandas as pd
import numpy as np

def energy_drift(df):
    E0 = df["total_energy"].iloc[0]
    return (df["total_energy"] - E0) / abs(E0)

def max_energy_error(df):
    return np.max(np.abs(energy_drift(df)))

def momentum_drift(df):
    P0 = np.sqrt(df["Px"].iloc[0]**2 + df["Py"].iloc[0]**2) + 1e-12
    P = np.sqrt(df["Px"]**2 + df["Py"]**2)
    return np.abs(P - P0) / P0