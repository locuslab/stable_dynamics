from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import odeint
from sympy import Dummy, lambdify, symbols
from sympy.physics import mechanics

CACHE = Path("pendulum-cache/")

# Modified from: http://jakevdp.github.io/blog/2017/03/08/triple-pendulum-chaos/
def pendulum_gradient(n, lengths=None, masses=1, friction=0.3):
    """Integrate a multi-pendulum with `n` sections"""
    #-------------------------------------------------
    # Step 1: construct the pendulum model
    
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
    P.set_vel(A, 0)

    # lists to hold particles, forces, and kinetic ODEs
    # for each pendulum in the chain
    particles = []
    forces = []
    kinetic_odes = []

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

        # Set forces & compute kinematic ODE
        forces.append((Pi, m[i] * g * A.x))
        # Add damping torque:
        forces.append((Ai, -1 * friction * u[i] * A.z))

        kinetic_odes.append(q[i].diff(t) - u[i])

        P = Pi

    # Generate equations of motion
    KM = mechanics.KanesMethod(A, q_ind=q, u_ind=u,
                               kd_eqs=kinetic_odes)
    fr, fr_star = KM.kanes_equations(particles, forces)

    #-----------------------------------------------------
    # Step 3: numerically evaluate equations and integrate

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
    kds = KM.kindiffdict()

    # substitute unknown symbols for qdot terms
    mm_sym = KM.mass_matrix_full.subs(kds).subs(unknown_dict)
    fo_sym = KM.forcing_full.subs(kds).subs(unknown_dict)

    # create functions for numerical calculation
    mm_func = lambdify(unknowns + parameters, mm_sym)
    fo_func = lambdify(unknowns + parameters, fo_sym)

    # function which computes the derivatives of parameters
    def gradient(y):
        rv = np.zeros_like(y)

        for i in range(y.shape[0]):
            # Assume in rad, rad/s:
            #y = np.concatenate([np.broadcast_to(initial_positions, n), np.broadcast_to(initial_velocities, n)])

            vals = np.concatenate((y[i,:], parameter_vals))
            sol = np.linalg.solve(mm_func(*vals), fo_func(*vals))
            rv[i,:] = np.array(sol).T[0]

        return rv

    # ODE integration
    return gradient

def _redim(inp):
    vec = np.array(inp)
    # Wrap all dimensions:
    n = vec.shape[1] // 2
    assert vec.shape[1] == n*2

    # Get angular positions:
    pos = vec[:,:n]
    l = 100

    if np.any(pos < -np.pi):
        # In multiples of 2pi
        adj, _ = np.modf((pos[pos < -np.pi] + np.pi) / (2*np.pi))
        # Scale it back
        pos[pos < -np.pi] = (adj * 2*np.pi) + np.pi
        assert not np.any(pos < -np.pi)

    if np.any(pos >= np.pi):
        # In multiples of 2pi
        adj, _ = np.modf((pos[pos >= np.pi] - np.pi) / (2*np.pi))
        # Scale it back
        pos[pos >= np.pi] = (adj * 2*np.pi) - np.pi
        assert not np.any(pos >= np.pi)

    vec[:,:n] = pos
    return vec

NUM_EXAMPLES = lambda n: 1000 * n
def build(props):
    # Number of joints in the pendulum:
    n = int(props["n"]) if "n" in props else 1
    test = "test" if "test" in props else "train"
    lowenergy =  "lowenergy" in props

    pen_gen = pendulum_gradient(n)
    le_str = "-lowenergy" if lowenergy else ""
    cache_path = CACHE / f"p-{n}{le_str}-{test}.npz"
    if not cache_path.exists():
        if lowenergy:
            X = np.zeros((NUM_EXAMPLES(n), 2 * n))
            # Pick values for displacement in range [-pi/4, pi/4] radians
            X[:,:n] = (np.random.rand(X.shape[0], n).astype(np.float32) - 0.5) * np.pi/2
        else:
            # Pick values in range [-pi, pi] radians, radians/sec
            X = (np.random.rand(NUM_EXAMPLES(n), n * 2).astype(np.float32) - 0.5) * 2 * np.pi
        Y = pen_gen(X)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, X=X, Y=Y)
    else:
        load = np.load(cache_path)
        X = load["X"]
        Y = load["Y"]

    rv = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(Y))
    rv._pendulum_gen = pen_gen
    rv._n = n
    rv._redim = _redim
    return rv

if __name__ == "__main__":
    # Continuous-time dynamics, learning function [x, x']_t -> [x', x'']_{t+1}
    def f_true(x):
        g_l = 9.81
        b = 0.3
        return np.array([x[:,1], -b*x[:,1] + g_l*np.sin(x[:,0]-np.pi)]).T

    gfunc = pendulum_gradient(1)
    X = (np.random.rand(10000, 2).astype(np.float32) - 0.5) * 2 * np.pi # Pick values in range [-pi, pi] radians, radians/sec

    test = gfunc(X)
    ref = f_true(X)

    differences = sorted(np.sum((test - ref) ** 2, axis=1))[-8:]
    print(differences)
    assert differences[-1] < 1e-8
