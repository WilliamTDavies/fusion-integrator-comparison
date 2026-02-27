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
    v_rot = np.array([p.vel[1], -p.vel[0]]) # Rotate velocity 90 degrees clockwise, valid for 2D orthogonal force directed out of the plane
    return p.charge * (E + B * v_rot)

def coulomb_force(p1, p2, k, eps):
    r_vec = p1.pos - p2.pos
    r = np.linalg.norm(r_vec)
    r_soft = np.sqrt(r*r + eps*eps) # Included to prevent singularity
    return k * p1.charge * p2.charge * r_vec / (r_soft**3) # Returns a force vector

# Acceleration
def compute_acceleration(particles, E, B, k, eps):
    acc = [np.zeros(2) for _ in particles]

    # Main calculation
    for i, p1 in enumerate(particles):
        if not p1.alive:
            continue

        F  = lorentz_force(p1, E, B)

        for j, p2 in enumerate(particles):
            if i !=j and p2.alive:
                F += coulomb_force(p1, p2, k, eps)

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
        momenta = [p.mass * p.vel for p in self.particles if p.alive]
        if not momenta: # Numerical robustness
            return np.zeros(2)
        return np.sum(momenta, axis=0)

    def total_energy(self):
        KE = sum(0.5 * p.mass * np.dot(p.vel, p.vel) for p in self.particles if p.alive)

        # Potential energy
        PE = 0.0
        for i, p1 in enumerate(self.particles):
            if not p1.alive:
                continue
            for j, p2 in enumerate(self.particles):
                if j > i and p2.alive:
                    r = np.linalg.norm(p1.pos - p2.pos)
                    r_soft = np.sqrt(r*r + self.eps*self.eps) # Included to prevent singularity
                    PE += self.k * p1.charge * p2.charge / r_soft

        return KE + PE
    
    # Collisions and fusion
    def handle_collision(self):
        for i, p1 in enumerate(self.particles):
            for j, p2 in enumerate(self.particles):
                if (j <= i) or (not (p1.alive and p2.alive)):
                    continue

                r = np.linalg.norm(p1.pos - p2.pos)

                # Collision logic
                if r < self.col_dist:
                    if self.should_fuse(p1, p2):
                        self.fuse(p1, p2)
                    else:
                        self.elastic_collision(p1, p2)

    def elastic_collision(self, p1, p2):
        r_vec = p2.pos - p1.pos
        r_norm = np.linalg.norm(r_vec)
        if r_norm == 0:
            return
        r_hat = r_vec / r_norm
        
        m1, m2 = p1.mass, p2.mass

        # Parallel components along collision axis
        v1_parallel = np.dot(p1.vel, r_hat) * r_hat
        v2_parallel = np.dot(p2.vel, r_hat) * r_hat

        # Perpendicular components
        v1_perp = p1.vel - v1_parallel
        v2_perp = p2.vel - v2_parallel

        # Elastic collision in 1D along r_hat
        v1_parallel_new = (v1_parallel*(m1 - m2) + 2*m2*v2_parallel) / (m1 + m2)
        v2_parallel_new = (v2_parallel*(m2 - m1) + 2*m1*v1_parallel) / (m1 + m2)

        # Update velocities
        p1.vel = v1_parallel_new + v1_perp
        p2.vel = v2_parallel_new + v2_perp

        # Stop potential multiple collisions 
        overlap = self.col_dist - np.linalg.norm(r_vec)
        if overlap > 0:
            correction = 0.5 * overlap * r_hat
            p1.pos -= correction
            p2.pos += correction

    def should_fuse(self, p1, p2):
        r = np.linalg.norm(p1.pos - p2.pos)
        r_soft = np.sqrt(r*r + self.eps*self.eps)
        U = self.k * p1.charge * p2.charge / r_soft
        v_rel = np.linalg.norm(p1.vel - p2.vel) + self.eps # Included to prevent singularity
        mu = p1.mass * p2.mass / (p1.mass + p2.mass)
        KE_rel = 0.5 * mu * (v_rel ** 2)

        if KE_rel > self.fuse_thresh * abs(U):
            return True
        else:
            return False # Could add stochastic fusion here if required, but invalidates some numerical methods

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
        self.handle_collision()
        self.time += dt