import numpy as np

import torch
from sympy import Dummy, lambdify, srepr, symbols
from sympy.core.function import AppliedUndef
from sympy.core.sympify import sympify
from sympy.physics import mechanics
from sympy.physics.vector import Vector
from sympy.printing.printer import Printer

from sympy2torch import sympy2torch

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

    gravity_direction = -A.x

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
    total_energy = (sum(gpe) + sum(ke)).subs(zip(parameters, parameter_vals))
    total_energy_func = sympy2torch(total_energy)
    # ODE integration
    return lambda inp: total_energy_func(**fixvalue(n, inp))

def fixvalue(n, value):
    keys = [f"q{i}_t" for i in range(n)] + [f"u{i}_t" for i in range(n)]
    rv = {}
    for i in range(2*n):
        if isinstance(value, np.ndarray):
            rv[keys[i]] = value[:,i]
        else:
            rv[keys[i]] = value[i]
    return rv

if __name__ == "__main__":
    # Test sympy2torch:
    n = 2
    efunc = pendulum_energy(n)

    # Test this!
    print(efunc(torch.tensor([0., 0., 0., 0.])))
    print(efunc(torch.tensor([ np.pi/2,  np.pi/2, 0., 0.])))
    print(efunc(torch.tensor([-np.pi/2, -np.pi/2, 0., 0.])))
    print(efunc(torch.tensor([ np.pi  ,  np.pi  , 0., 0.])))

    print(efunc(torch.tensor([0., 0., 1., 0.])))
    print(efunc(torch.tensor([0., 0., -1., 0.])))

    print(efunc(torch.tensor([0., 0., 1., 1.])))
    print(efunc(torch.tensor([0., 0., -1., -1.])))

    # assert pendulum_energy([0., 0. ])

    # q0(t)) + g*m1*(-l0*sin(q0(t)) - l1*sin(q1(t)))
    # l0**2*m0*u0(t)**2/2 + l0**2*m1*u0(t)**2/2 + l0*l1*m1*(sin(q0(t))*sin(q1(t)) + cos(q0(t))*cos(q1(t)))*u0(t)*u1(t) + l1**2*m1*u1(t)**2/2
