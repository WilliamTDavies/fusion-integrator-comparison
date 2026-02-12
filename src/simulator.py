import numpy as np

# Particle definition
class Particle:
    def __init__ (self, pid, position, velocity, mass, charge):
        self.id = pid
        self.pos = position
        self.vel = velocity
        self.mass = mass
        self.charge = charge
        self.alive = True

# Force models
def lorentz_force(p, E, B):
    return  p.charge * (E + np.cross(p.vel, B))

def coulomb_force(p1, p2, k, eps):
    r = np.linalg.norm(p1.pos - p2.pos) + eps # Included to prevent singularity
    return k * p1.charge * p2.charge

# Acceleration
def compute_acceleration(particles, E, B):
    acc = [np.zeros[3] for _ in particles]

    # Main calculation
    for i, p1 in enumerate(particles):
        if not p1.alive:
            continue

        F  = lorentz_force(p1, E, B)

        for j, p2 in enumerate(particles):
            if i !=j and p2.alive:
                F += coulomb_force(p1, p2)

        acc[i] = F / p1.mass

    return acc

# Plasma system
class System:
    def __init__(self, particles, E, B, col_dist, fuse_thresh, eps, k):
        self.particles = particles
        self.E = E
        self.B = B
        self.col_dist = col_dist
        self.fuse_thresh = fuse_thresh
        self.eps = eps
        self.k = k
        self.time = 0.0
    
    def total_momentum(self):
        return sum(p.vel * p.mass for p in self.particles if p.alive)

    def total_energy(self):
        KE = sum(0.5 * p.mass * np.dot(p.vel, p.vel) for p in self.particles if p.alive)

        # Potential energy
        PE = 0.0
        for i, p1 in enumerate(self.particles):
            if not p1.alive:
                continue
            for j, p2 in enumerate(self.particles):
                if j > i and p2.alive:
                    r = np.linalg.norm(p1.pos - p2.pos) + self.eps # Included to prevent singularity
                    PE += self.k * p1.charge * p2.charge / r

        return KE + PE
    
    # Collisions and fusion
    def handle_collision(self):
        pass

    def elastic_collision(self, p1, p2):
        pass

    def should_fuse(self, p1, p2):
        pass

    def fuse(self, p1, p2):
        pass

    # Update system
    def step_update(self, integrator, dt):
        integrator.step(self, dt)
        self.handle_collision(self)
        self.time += dt