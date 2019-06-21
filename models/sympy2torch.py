import numpy as np

import torch
from sympy import Dummy, lambdify, srepr, symbols
from sympy.core.function import AppliedUndef
from sympy.core.sympify import sympify
from sympy.physics import mechanics
from sympy.physics.vector import Vector
from sympy.printing.printer import Printer

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
    test_sympy2torch()
