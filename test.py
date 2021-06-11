import numpy as np
from dezero.utils import plot_dot_graph
from dezero import *

x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
y = x0+x1
txt = plot_dot_graph(y)
