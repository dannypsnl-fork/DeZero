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
        ys = add(x0, x1)
        expected = np.array(5)
        self.assertEqual(ys.data, expected)

    def test_add_backward(self):
        x = Variable(np.array(3.0))
        y = add(x, x)
        y.backward()
        expected = 2
        self.assertEqual(x.grad, expected)

    def test_add_backward2(self):
        x = Variable(np.array(3.0))
        y = add(add(x, x), x)
        y.backward()
        expected = 3
        self.assertEqual(x.grad, expected)

    def test_cleargrad(self):
        x = Variable(np.array(3.0))
        y = add(x, x)
        y.backward()
        expected = 2
        self.assertEqual(x.grad, expected)
        x.cleargrad()
        y = add(add(x, x), x)
        y.backward()
        expected = 3
        self.assertEqual(x.grad, expected)


if __name__ == '__main__':
    unittest.main()
