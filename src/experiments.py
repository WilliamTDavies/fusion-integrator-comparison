import pandas as pd
import time
import numpy as np
from pathlib import Path
from copy import deepcopy

from simulator import Particle, System
from integrators import EulerIntegrator, RK4Integrator, VelocityVerletIntegrator

def particle_initialisation(p_num, min_dist=0.2, max_attempts=1000):
    p_list = []

    for i in range(p_num):
        attempts = 0

        while True:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError("Could not place particles without overlap. Reduce min_dist or particle count.")

            pos = np.random.uniform(-2.0, 2.0, 2)

            # Check distance to all existing particles
            too_close = False
            for p in p_list:
                if np.linalg.norm(pos - p.pos) < min_dist:
                    too_close = True
                    break

            if not too_close:
                break  # Accept this position

        vel = np.random.uniform(-0.25, 0.25, 2)
        m = 1.0
        q = np.random.choice([-1.0, 1.0])

        p = Particle(i, pos, vel, m, q)
        p_list.append(p)

    return p_list

def full_simulation(int_method, system, dt, T_max):
    # Integrator selection
    integrator_dict = {
        "euler": EulerIntegrator(),
        "rk4": RK4Integrator(),
        "verlet": VelocityVerletIntegrator()
    }
    integrator = integrator_dict[int_method]

    # Paths
    ROOT = Path(__file__).resolve().parent.parent
    DATA = ROOT / "results" / "data"
    DATA.mkdir(parents=True, exist_ok=True)

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
    df.to_csv(DATA / filename, index=False)
    return runtime

def system_snapshot(system):
    # System variables
    row = {
        "t": round(system.time, 10), # Prevents clustering and is at an acceptable resolution that we can discount possible floating point errors
        "total_energy": system.total_energy()
    }
    P = system.total_momentum()
    row["Px"] = P[0]
    row["Py"] = P[1]

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
    np.random.seed(42)
    dt_grid =  [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1e-2]
    T_max = 10.0
    p_list = particle_initialisation(10)
    runtime_records = []
    # Run experiments
    for method in ["euler", "rk4", "verlet"]:
        for dt in dt_grid:
            print(f"Running: {method} with dt={dt}")
            
            # Initialise experimental system
            system = System(
                deepcopy(p_list),       # Particles
                [0.0, 0.0],             # Electric Field
                0.0,                    # Magnetic field
                0.2,                    # Collision distance
                0.75,                   # Fusion threshold
                5e-2,                   # Epsilon
                1.0,                    # Coulomb constant
                enable_collisions=True,
                enable_fusion=False
            )
            runtime = full_simulation(method, system, dt, T_max)
            runtime_records.append({
                "integrator_method": method,
                "dt" : dt,
                "runtime_secs": runtime
            })

    # Paths
    ROOT = Path(__file__).resolve().parent.parent
    DATA = ROOT / "results" / "data"
    DATA.mkdir(parents=True, exist_ok=True)

    # Save data
    runtime_df = pd.DataFrame(runtime_records)
    runtime_df.to_csv(DATA / "runtime_data.csv", index=False)

    print("All experiments complete")

# Run the experiments
if __name__ == "__main__":
    sweep()