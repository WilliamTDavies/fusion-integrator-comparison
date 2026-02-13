import pandas as pd
import numpy as np

def energy_drift(df):
    return abs(df["total_energy"] - df["total_energy"].iloc[0]) / abs(df["total_energy"].iloc[0])

def max_energy_error(df):
    return energy_drift(df).max()