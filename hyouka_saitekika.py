import math
import numpy as np

from scipy.optimize import differential_evolution
from MRR.SHADE_before_heiretu import SHADE
from MRR.benchmark_function import BenchmarkFunction as BF

bf = BF()

#Sphere
a = 5.12
bounds = np.array((-a,a))
params = 0
pop_size = 10
max_iter = 6000
H = 50
tol = 0.01

SHADE(bf.sphere, bounds, params, pop_size, max_iter, H, tol, callback = None, rng = None)
