import unittest

import numpy as np

from dezero import *


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_add(self):
        x0 = Variable(np.array(2))
        x1 = Variable(np.array(3))
        ys = x0 + x1
        expected = np.array(5)
        self.assertEqual(ys.data, expected)

    def test_add_backward(self):
        x = Variable(np.array(3.0))
        y = x + x
        y.backward()
        expected = 2
        self.assertEqual(x.grad, expected)

    def test_add_backward2(self):
        x = Variable(np.array(3.0))
        y = x + x + x
        y.backward()
        expected = 3
        self.assertEqual(x.grad, expected)

    def test_cleargrad(self):
        x = Variable(np.array(3.0))
        y = x + x
        y.backward()
        expected = 2
        self.assertEqual(x.grad, expected)
        x.cleargrad()
        y = x + x + x
        y.backward()
        expected = 3
        self.assertEqual(x.grad, expected)

    def test_graph(self):
        x = Variable(np.array(2.0))
        a = square(x)
        y = square(a) + square(a)
        y.backward()
        expected = 64
        self.assertEqual(x.grad, expected)

    def test_mul(self):
        a = Variable(np.array(3.0))
        b = Variable(np.array(2.0))
        c = Variable(np.array(1.0))
        y = a * b + c
        y.backward()
        self.assertEqual(a.grad, 2)
        self.assertEqual(b.grad, 3)

    def test_sphere(self):
        def sphere(x, y):
            z = x ** 2 + y ** 2
            return z
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = sphere(x, y)
        z.backward()
        self.assertEqual(x.grad, 2.0)
        self.assertEqual(y.grad, 2.0)


if __name__ == '__main__':
    unittest.main()
