import numpy as np
from copy import deepcopy
from simulator import compute_accelerations

class Integrator:
    def step(self, system, dt):
        raise NotImplementedError # Ensures soft crashes
    
class EulerIntegrator(Integrator):
    def step(self, system, dt):
        pass

class RK4Integrator(Integrator):
    def step(self, system, dt):
        pass

class VelocityVerletIntegrator(Integrator):
    def step(self, system, dt):
        pass