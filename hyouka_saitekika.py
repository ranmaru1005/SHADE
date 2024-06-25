import math
import numpy as np

from scipy.optimize import differential_evolution
from MRR.SHADE import SHADE
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
rng = np.random.Generator

print( SHADE(bf.rastrigin, bounds, params, pop_size, max_iter, H, tol, callback = None, rng = None) )

result = differential_evolution(bf.rastrigin, 
                            bounds, 
                            args=(params,), 
                            strategy="currenttobest1bin", 
                            workers=-1, 
                            updating="deferred", 
                            popsize=15,
                            maxiter=500,
                            seed=rng)

print("E = ",result.fun)
print("X(解) = ",result.x)
