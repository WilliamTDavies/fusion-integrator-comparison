import pandas as pd
import time
import numpy as np
from pathlib import Path

from simulator import Particle, System
from integrators import EulerIntegrator, RK4Integrator, VelocityVerletIntegrator

def initial_conditions(p_num, E, B, col_dist, fuse_thresh, eps, k):
    # Particle initialisation
    p_list = []
    for i in range(p_num):
        pos = np.random.uniform(-1.0, 1.0, 2)
        vel = np.random.uniform(-0.25, -0.25, 2)
        m = 1
        q = np.random.choice(-1.0, 1.0) # Only positive and negative integer charges
        p = Particle(i, pos, vel, m, q)
        p_list.append(p)

    # Create the specific system
    system = System(p_list, E, B, col_dist, fuse_thresh, eps, k)
    return system

def full_simulation(int_method, system, dt, T_max):
    # Integrator selection
    integrator_dict = {
        "euler": EulerIntegrator(),
        "rk4": RK4Integrator(),
        "verlet": VelocityVerletIntegrator()
    }
    integrator = integrator_dict[int_method]

    # Paths
    ROOT = Path(__file__).resolve.parent().parent()
    DATA = ROOT / "results" / "data"

    # Generate data
    records = []
    records.append(system_snapshot(system))
    start_time = time.perf_counter()
    while system.time < T_max:
        system.step_update(integrator, dt)
        records.append(system_snapshot(system))
    runtime = time.perf_counter() - start_time

    # Save data
    df = pd.DataFrame.from_records(records)
    filename = f"{int_method}_{dt:.0e}.csv"
    df.to_csv(DATA/ filename, index=False)
    return runtime

def system_snapshot(system):
    # System variables
    row = {
        "t": system.time,
        "total_energy": system.total_energy()
    }
    P = system.total_momentum()
    row["Px"] = P[0]
    row["Py"] = P[1]
    row["Pz"] = P[2]

    # Particle variables
    for p in system.particles:
        row[f"x_{p.id}"] = p.pos[0]
        row[f"y_{p.id}"] = p.pos[1]
        row[f"vx_{p.id}"] = p.vel[0]
        row[f"vy_{p.id}"] = p.vel[1]
        row[f"alive_{p.id}"] = int(p.alive)

    return row

def sweep():
    # Initialisation
    np.random.seed(0)
    dt_grid = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4] 
    T_max = 1.0
    runtime_records = []
    # Run experiments
    for method in ["euler", "rk4", "verlet"]:
        for dt in dt_grid:
            print(f"Running: {method} with dt={dt}")
            system = initial_conditions(3, [0.0, 0.0], 0.1, 0.05, 0.5, 1e-12, 1.0) # Number of particles, electric field, magnetic field, collision distance, fusion threshold, epsilon, coulomb constant
            runtime = full_simulation(method, system, dt, T_max)
            runtime_records.append({
                "integrator_method": method,
                "dt" : dt,
                "runtime_secs": runtime
            })

    # Paths
    ROOT = Path(__file__).resolve.parent().parent()
    DATA = ROOT / "results" / "data"

    # Save data
    runtime_df = pd.DataFrame(runtime_records)
    runtime_df.to_csv(DATA / "runtime_data.csv", index=False)

    print("All experiments complete")