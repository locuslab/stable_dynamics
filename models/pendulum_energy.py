import numpy as np
from scipy.integrate import odeint

import matplotlib.pyplot as plt
import torch
from sympy import Dummy, lambdify, srepr, symbols
from sympy.core.function import AppliedUndef
from sympy.core.sympify import sympify
from sympy.physics import mechanics
from sympy.physics.vector import Vector
from sympy.printing.printer import Printer


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
    print(srepr(gpe[0].subs(zip(parameters, parameter_vals))))
    total_gpe = sum(gpe)
    print(total_gpe)

    print(sympy2torch(gpe[0]))

    print(srepr(ke[0].subs(zip(parameters, parameter_vals))))
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


class TorchPrinter(Printer):
    printmethod = "_torchrepr"
    _default_settings = {
        "order": None
    }

    def _print_Add(self, expr, order=None):
        terms = list(map(self._print, self._as_ordered_terms(expr, order=order)))
        def __inner_sum(**kw):
            x = terms[0](**kw)
            for r in terms[1:]:
                x = torch.add(x, r(**kw))
            return x
        return __inner_sum

    def _print_Function(self, expr):
        __FUNCTION_MAP = { "sin": torch.sin, "cos": torch.cos, "tan": torch.tan }

        if expr.func.__name__ in __FUNCTION_MAP:
            func = __FUNCTION_MAP[expr.func.__name__]
            args = [self._print(a) for a in expr.args]
            return lambda **kw: func(*(a(**kw) for a in args))
        else:
            key = f"{expr.func.__name__}_{', '.join([str(a) for a in expr.args])}"
            return lambda **kw: kw[key]

    def _print_Half(self, expr):
        return lambda **kw: 0.5

    def _print_Integer(self, expr):
        return lambda **kw: expr.p

    def _print_NaN(self, expr):
        return lambda **kw: float('nan')

    def _print_Mul(self, expr, order=None):
        assert len(expr.args) > 1
        terms = [self._print(a) for a in expr.args]

        def __inner_mul(**kw):
            x = terms[0](**kw)
            for r in terms[1:]:
                x = torch.mul(x, r(**kw))
            return x
        return __inner_mul

    def _print_Rational(self, expr):
        return lambda **kw: expr.p/expr.q

    def _print_Fraction(self, expr):
        numer = self._print(expr.numerator)
        denom = self._print(expr.denominator)
        return lambda **kw: numer(**kw), denom(**kw)

    def _print_Float(self, expr):
        return lambda **kw: float(expr.evalf())

    def _print_Symbol(self, expr):
        d = expr._assumptions.generator
        # print the dummy_index like it was an assumption
        if expr.is_Dummy or d != {}:
            raise NotImplementedError()

        return lambda **kw: kw[expr.name]

    def _print_FracElement(self, frac):
        numer_terms = list(frac.numer.terms())
        assert len(numer_terms) == 1
        #numer_terms.sort(key=frac.field.order, reverse=True)
        denom_terms = list(frac.denom.terms())
        assert len(denom_terms) == 1
        #denom_terms.sort(key=frac.field.order, reverse=True)
        numer = self._print(numer[0])
        denom = self._print(denom[0])
        return lambda *kw: torch.div(numer(*kw), denom(*kw))

def sympy2torch(expr):
    return TorchPrinter()._print(expr)

def test_sympy2torch():
    def test(expr, vars, examples):
        func = sympy2torch(sympify(expr))
        for i, (inp, out) in enumerate(examples):
            kw = dict(zip(vars, torch.tensor(inp)))
            v = func(**kw)
            if v != out:
                print(f"FAILED\t{expr}@{i}\t{kw}\t Expected({out}) != Received({v})")

    test("Mul(Integer(2), Symbol('x'))", "x", [([0], 0), ([1], 2), ([2], 4)])
    test("Add(Integer(2), Symbol('y'))", "xy", [([0, 0], 2), ([1, 0], 2), ([2, 2], 4), ([2, 1], 3)])
    test("Mul(Add(Integer(2), Symbol('y')), Symbol('x'))", "xy", [([0, 0], 0), ([1, 0], 2), ([2, 0], 4), ([2, 1], 6)])
    test("Mul(Integer(-1), Float('9.8100000000000005', precision=53), sin(Function('q0')(Symbol('t'))))", ["q0_t"], [([0.], 0), ([np.pi/2], -9.81), ([np.pi/6], -9.81*0.5), ([np.pi/4], -9.81*(0.5**0.5))])

    print("Completed!")

if __name__ == "__main__":
    # Test sympy2torch:
    n = 2
    efunc = pendulum_energy(n)

    # q0(t)) + g*m1*(-l0*sin(q0(t)) - l1*sin(q1(t)))
    # l0**2*m0*u0(t)**2/2 + l0**2*m1*u0(t)**2/2 + l0*l1*m1*(sin(q0(t))*sin(q1(t)) + cos(q0(t))*cos(q1(t)))*u0(t)*u1(t) + l1**2*m1*u1(t)**2/2

    vars = np.random.rand(1, n*2)
    # print(efunc(vars))
    test_sympy2torch()
