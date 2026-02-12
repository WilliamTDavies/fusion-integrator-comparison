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

        s0 = deepcopy(system.particles) # Work on copies to prevent accidental data deletion

        a1 = compute_accelerations(s0, system.E, system.B)
        s1 = deepcopy(s0)
        for p, a in zip(s1.particles, a1):
            p.pos += p.vel * dt * 0.5
            p.vel += a * dt * 0.5

        a2 = compute_accelerations(s1, system.E, system.B)
        s2 = deepcopy(s0)
        for p, a in zip(s2.particles, a2):
            p.pos += p.vel * dt * 0.5
            p.vel += a * dt * 0.5

        a3 = compute_accelerations(s2, system.E, system.B)
        s3 = deepcopy(s0)
        for p, a in zip(s3.particles, a3):
            p.pos += p.vel * dt
            p.vel += a * dt
        
        a4 = compute_accelerations(s3)

        for i, p in enumerate(system.particles):
            if not p.alive:
                continue
            p.pos += (s0[i].vel + 2 * s1[i].vel + 2 * s2[i].vel + s3[i].vel) * dt / 6
            p.vel += (a1[i] + 2 * a2[i] + 2 * a3[i] + a4[i]) * dt / 6

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