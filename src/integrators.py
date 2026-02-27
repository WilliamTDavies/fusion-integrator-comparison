import numpy as np
from copy import deepcopy
from simulator import compute_acceleration

class Integrator:
    def step(self, system, dt):
        raise NotImplementedError # Ensures soft crashes
    
class EulerIntegrator(Integrator):
    def step(self, system, dt):
        acc = compute_acceleration(system.particles, system.E, system.B, system.k, system.eps)

        for p, a in zip(system.particles, acc):
            if not p.alive:
                continue

            p.pos += p.vel * dt
            p.vel += a * dt

class RK4Integrator(Integrator):
    def step(self, system, dt):

        s0 = deepcopy(system.particles) # Copy to prevent accidental deletion

        # Stage 1
        a1 = compute_acceleration(s0, system.E, system.B, system.k, system.eps)
        s1 = deepcopy(s0)
        for i, p in enumerate(s1):
            if not p.alive:
                continue
            p.pos += s0[i].vel * (dt / 2)
            p.vel += a1[i] * (dt / 2)

        # Stage 2
        a2 = compute_acceleration(s1, system.E, system.B, system.k, system.eps)
        s2 = deepcopy(s0)
        for i, p in enumerate(s2):
            if not p.alive:
                continue
            p.pos += s1[i].vel * (dt / 2)
            p.vel += a2[i] * (dt / 2)

        # Stage 3
        a3 = compute_acceleration(s2, system.E, system.B, system.k, system.eps)
        s3 = deepcopy(s0)
        for i, p in enumerate(s3):
            if not p.alive:
                continue
            p.pos += s2[i].vel * dt
            p.vel += a3[i] * dt

        # Stage 4
        a4 = compute_acceleration(s3, system.E, system.B, system.k, system.eps)

        # RK4 Update
        for i, p in enumerate(system.particles):
            if not p.alive:
                continue

            p.pos += (
                s0[i].vel
                + 2 * s1[i].vel
                + 2 * s2[i].vel
                + s3[i].vel
            ) * dt / 6

            p.vel += (
                a1[i]
                + 2 * a2[i]
                + 2 * a3[i]
                + a4[i]
            ) * dt / 6

class VelocityVerletIntegrator(Integrator):
    def step(self, system, dt):
        # Position
        acc = compute_acceleration(system.particles, system.E, system.B, system.k, system.eps)
        for p, a in zip(system.particles, acc):
            if not p.alive:
                continue
            p.pos += p.vel * dt + 0.5 * a * dt ** 2

        # Velocity
        acc_new = compute_acceleration(system.particles, system.E, system.B, system.k, system.eps)
        for p, a0, a1 in zip(system.particles, acc, acc_new):
            if not p.alive:
                continue
            p.vel += 0.5 * (a0 + a1) * dt