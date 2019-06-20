import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import odeint
from sympy import Dummy, lambdify, symbols
from sympy.physics import mechanics
from sympy.physics.vector import Vector

def pendulum_energy(n=1, lengths=1, masses=1):
    # Generalized coordinates and velocities
    # (in this case, angular positions & velocities of each mass) 
    q = mechanics.dynamicsymbols('q:{0}'.format(n))
    u = mechanics.dynamicsymbols('u:{0}'.format(n))

    # mass and length
    m = symbols('m:{0}'.format(n))
    l = symbols('l:{0}'.format(n))

    # gravity and time symbols
    g, t = symbols('g,t')

    #--------------------------------------------------
    # Step 2: build the model using Kane's Method

    # Create pivot point reference frame
    A = mechanics.ReferenceFrame('A')
    P = mechanics.Point('P')
    Origin = P
    P.set_vel(A, 0)

    gravity_direction = -A.y

    # lists to hold particles, forces, and kinetic ODEs
    # for each pendulum in the chain
    particles = []
    forces = []

    gpe = []
    ke = []

    cartVel = 0.0
    cartPos = 0.0

    for i in range(n):
        # Create a reference frame following the i^th mass
        Ai = A.orientnew('A' + str(i), 'Axis', [q[i], A.z])
        Ai.set_ang_vel(A, u[i] * A.z)

        # Create a point in this reference frame
        Pi = P.locatenew('P' + str(i), l[i] * Ai.x)
        Pi.v2pt_theory(P, A, Ai)

        # Create a new particle of mass m[i] at this point
        Pai = mechanics.Particle('Pa' + str(i), Pi, m[i])
        particles.append(Pai)

        # Calculate the cartesian position and velocity:
        # cartPos += l[i] * q[i]
        pos = Pi.pos_from(Origin)

        ke.append(Pai.kinetic_energy(A))
        gpe.append(m[i] * g * (Pi.pos_from(Origin) & gravity_direction))

        P = Pi


    # lengths and masses
    if lengths is None:
        lengths = np.ones(n) / n
    lengths = np.broadcast_to(lengths, n)
    masses = np.broadcast_to(masses, n)

    # Fixed parameters: gravitational constant, lengths, and masses
    parameters = [g] + list(l) + list(m)
    parameter_vals = [9.81] + list(lengths) + list(masses)

    # define symbols for unknown parameters
    unknowns = [Dummy() for i in q + u]
    unknown_dict = dict(zip(q + u, unknowns))

    # create functions for numerical calculation
    total_gpe = sum(gpe)
    print(total_gpe)
    total_ke = sum(ke)
    print(total_ke)

    total_energy_func = lambdify(unknowns + parameters,  total_gpe + total_ke)

    # function which computes the derivatives of parameters
    def total_energy(y):
        rv = np.zeros((y.shape[0],))

        for i in range(y.shape[0]):
            # Assume in rad, rad/s:
            #y = np.concatenate([np.broadcast_to(initial_positions, n), np.broadcast_to(initial_velocities, n)])

            vals = np.concatenate((y[i,:], parameter_vals))
            rv[i] = total_energy_func(*vals)

        return rv

    # ODE integration
    return total_energy

if __name__ == "__main__":
    n = 2
    efunc = pendulum_energy(n)

    vars = np.random.rand(1, n*2)
    print(efunc(vars))
