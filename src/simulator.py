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
    v_rot = np.array([p.vel[1], -p.vel[0]]) # Rotate velocity 90 degrees clockwise
    return p.charge * (E + B * v_rot)

def coulomb_force(p1, p2, k, eps):
    r = np.linalg.norm(p1.pos - p2.pos) + eps # Included to prevent singularity
    return k * p1.charge * p2.charge

# Acceleration
def compute_acceleration(particles, E, B):
    acc = [np.zeros[2] for _ in particles]

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
        for i, p1 in enumerate(self.particles):
            for j, p2 in enumerate(self.particles):
                if (j <= i) or (not (p1.alive and p2.alive)):
                    continue

                r = np.linalg.norm(p1.pos - p2.pos) + self.eps # Included to prevent singularity

                # Collision logic
                if r < self.col_dist:
                    if self.should_fuse(self, p1, p2):
                        self.fuse(self, p1, p2)
                    else:
                        self.elastic_collision(self, p1, p2)

    def elastic_collision(self, p1, p2):
        m1, m2 = p1.mass, p2.mass
        v1, v2 = m1.vel, p2.vel

        p1.vel = (v1 * (m1 - m2) + 2 * v2 * m2) / (m1 + m2)
        p2.vel = (v2 * (m2 - m1) + 2 * v1 * m1) / (m1 + m2)

    def should_fuse(self, p1, p2):
        r = np.linalg.norm(p1.pos - p2.pos) + self.eps # Included to prevent singularity
        U = self.k * p1.charge * p2.charge / r
        v_rel = np.linalg.norm(p1.vel - p2.vel) + self.eps # Included to prevent singularity
        mu = p1.mass * p2.mass / (p1.mass + p2.mass)
        KE_rel = 0.5 * mu * (v_rel ** 2)

        if KE_rel > self.fuse_thresh * U:
            return True
        else:
            tunnel_prob = np.exp(-U/KE_rel)
            return np.random.rand() < tunnel_prob

    def fuse(self, p1, p2):
        new_pid = 1 + len(self.particles)
        new_mass = p1.mass + p2.mass
        new_charge = p1.charge + p2.charge
        new_pos = (p1.mass * p1.pos + p2.mass * p2.pos) / new_mass
        new_vel = (p1.mass * p1.vel + p2.mass * p2.vel) / new_mass

        p_new = Particle(new_pid, new_pos, new_vel, new_mass, new_charge)
        self.particles.append(p_new)
        p1.alive = False
        p2.alive = False

    # Update system
    def step_update(self, integrator, dt):
        integrator.step(self, dt)
        self.handle_collision(self)
        self.time += dt