import numpy as np
from copy import deepcopy
from simulator import compute_accelerations

class Integrator:
    def step(self, system, dt):
        raise NotImplementedError # Ensures soft crashes
    
class EulerIntegrator(Integrator):
    def step(self, system, dt):
        acc = compute_accelerations(system.particles, system.E, system.B)

        for p, a in zip(system.particles, acc):
            if not p.alive:
                continue

            p.pos += p.vel * dt
            p.vel += a * dt

class RK4Integrator(Integrator):
    def step(self, system, dt):
        pass

class VelocityVerletIntegrator(Integrator):
    def step(self, system, dt):
        # Position
        acc = compute_accelerations(system.particles, system.E, system.B)
        for p, a in zip(system.particles, acc):
            if not p.alive:
                continue
            p.pos += p.vel * dt + 0.5 * a * dt ** 2
        
        # Velocity
        acc_new = compute_accelerations(system.particles, system.E, system.B)
        for p, a0, a1 in zip(system.particles, acc_new):
            if not p.alive:
                continue
            p.vel += np.mean(a0, a1) * dt
