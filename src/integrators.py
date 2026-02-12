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
        pass
