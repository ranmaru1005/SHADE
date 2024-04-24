from re import A
import numpy as np
from typing import Any, Dict, List

class TransferFunction:
    """Simulator of the transfer function of the MRR filter.
    Args:
        L (List[float]): List of the round-trip length.
        K (List[float]): List of the coupling rate.
        config (Dict[str, Any]): Configuration of the MRR.
            Keys:
                eta (float): The coupling loss coefficient.
                n_eff (float): The effective refractive index.
                n_g (float): The group index.
                alpha (float): The propagation loss coefficient.
    Attributes:
        L (List[float]): List of the round-trip length.
        K (List[float]): List of the coupling rate.
        eta (float): The coupling loss coefficient.
        n_eff (float): The effective refractive index.
        n_g (float): The group index.
        a (List[float]): List of the propagation loss.
    """

    def __init__(
        self,
        L,
        K,
        config: Dict[str, Any]
    ) -> None:
        self.L: List[float] = L
        self.K: List[float] = K
        self.center_wavelength: float = config['center_wavelength']
        self.eta: float = config['eta']
        self.n_eff: float = config['n_eff']
        self.n_g: float = config['n_g']
        self.a: List[float] = np.exp(- config['alpha'] * L)

    def _C(self, K_k: float) -> np.array:
        return 1 / (-1j * self.eta * np.sqrt(K_k)) * np.array([
            [1, - self.eta * np.sqrt(self.eta - K_k)],
            [np.sqrt(self.eta - K_k) * self.eta, - self.eta ** 2]
        ])

    def _R(self, a_k: float, L_k: float, l: np.array) -> np.array:
        x = 1j * np.pi * L_k * self.n_g * (l - self.center_wavelength) / self.center_wavelength / self.center_wavelength
        return np.array([
            [np.exp(x) / np.sqrt(a_k), 0],
            [0, np.exp(-x) * np.sqrt(a_k)]
        ], dtype="object")

    def _reverse(self, arr: List) -> List:
        return arr[::-1]

    def _M(self, l: List[float]) -> np.array:
        product = np.identity(2)
        for _K, _a, _L in zip(
            self._reverse(self.K[1:]),
            self._reverse(self.a),
            self._reverse(self.L)
        ):
            product = np.dot(product, self._C(_K))
            product = np.dot(product, self._R(_a, _L, l))
        product = np.dot(product, self._C(self.K[0]))
        return product

    def _D(self, l: List[float]) -> np.mat:
        return 1 / self._M(l)[0, 0]

    def print_parameters(self) -> None:
        print('eta:', self.eta)
        print('center_wavelength:', self.center_wavelength)
        print('n_eff:', self.n_eff)
        print('n_g:', self.n_g)
        print('a:', self.a)
        print('K:', self.K.tolist())
        print('L:', self.L.tolist())

    def simulate(self, l: List[float]) -> np.array:
        y = 20 * np.log10(np.abs(self._D(l)))
        return y.reshape(y.size)

class build_TransferFunction_Factory:
    def __init__(self, config):
        self.config = config

    def create(self, L, K):
        return TransferFunction(L, K, self.config)

def build_TransferFunction(config):
    """Partial-apply config to TransferFunction
    Args:
        config (Dict[str, Any]): Configuration of the TransferFunction
    Returns:
        TransferFunction_with_config: TransferFunction that is partial-applied config to.
    """
    return build_TransferFunction_Factory(config).create