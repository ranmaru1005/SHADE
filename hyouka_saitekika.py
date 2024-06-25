import math
import numpy as np
import numpy.typing as npt

from scipy.optimize import differential_evolution
from MRR.SHADE import SHADE
from MRR.benchmark_function import BenchmarkFunction as BF

bf = BF()

class OptimizeKParams:
    L: npt.NDArray[np.float_]
    n_g: float
    n_eff: float
    eta: float
    alpha: float
    center_wavelength: float
    length_of_3db_band: float
    FSR: np.float_
    max_crosstalk: float
    H_p: float
    H_s: float
    H_i: float
    r_max: float
    weight: list[float]





#Sphere
a = 5.12
bounds = np.array((-a,a))
params = 0
pop_size = 10
max_iter = 6000
H = 50
tol = 0.01
rng = np.random.Generator




#print( SHADE(bf.rastrigin, bounds, params, pop_size, max_iter, H, tol, callback = None, rng = None) )

result = differential_evolution(bf.rastrigin, 
                            bounds, 
                            strategy="currenttobest1bin", 
                            workers=-1, 
                            updating="deferred", 
                            popsize=15,
                            maxiter=500,
                            seed=rng)

print("E = ",result.fun)
print("X(解) = ",result.x)
