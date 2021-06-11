import numpy as np

from .core import Variable
from .core import Function
from .core import as_array


class Exp(Function):
    def forward(self, xs):
        x = xs
        return np.exp(x)

    def backward(self, gy):
        x = self.inputs[0].data
        return np.exp(x) * gy


def exp(x):
    return Exp()(x)


class Square(Function):
    def forward(self, xs):
        x = xs
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0].data
        return 2 * x * gy


def square(x):
    return Square()(x)


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return (y,)

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)


class Mul(Function):
    def forward(self, x0, x1):
        return x0 * x1

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


def mul(x0, x1):
    return Mul()(x0, x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


def neg(x):
    return Neg()(x)


class Sub(Function):
    def forward(self, x0, x1):
        y = x0-x1
        return y

    def backward(self, gy):
        return gy, -gy


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)


class Div(Function):
    def forward(self, x0, x1):
        y = x0/x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy/x1
        gx1 = gy*(-x0 / x1**2)
        return gx0, gx1


def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x**self.c
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        gx = c*x**(c-1) * gy
        return gx


def pow(x, c):
    return Pow(c)(x)


Variable.__pow__ = pow
Variable.__truediv__ = div
Variable.__rtruediv__ = rdiv
Variable.__sub__ = sub
Variable.__rsub__ = rsub
Variable.__neg__ = neg
Variable.__mul__ = mul
Variable.__rmul__ = mul
Variable.__add__ = add
Variable.__radd__ = add
