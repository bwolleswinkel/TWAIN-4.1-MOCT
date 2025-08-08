"""This script is used for some utilities

"""

from typing import TypeVar, TypeAlias

import numpy as np
import numpy.typing as npt

# ------ TYPE ALIASES ------

T = TypeVar('T', int, float, complex, bool)
NPArray: TypeAlias = npt.NDArray[np.dtype[T]]

# ------ FUNCTIONS ------


def remap(a: NPArray[float], range: tuple[float, float]) -> NPArray[float]:
    return range[0] + (range[1] - range[0]) * (a - np.min(a)) / (np.max(a) - np.min(a))


def ensure_list(obj) -> list:
    match obj:
        case list():
            return obj
        case tuple():
            return list(obj)
        case np.ndarray():
            return obj.tolist()
        case float() | int() | bool():
            return [obj]
        case _:
            return ValueError(f"Unsupported type '{type(obj)}'")
        

def find_datum(array: NPArray[float]) -> list[float, float]:
    #: Round smallest element down to the nearest multiple of 1000
    return np.floor(np.min(array, axis=0) / 1000) * 1000
        

def devisors(n: int) -> NPArray[int]:
    # FROM: https://stackoverflow.com/questions/171765/what-is-the-best-way-to-get-all-the-divisors-of-a-number
    divs = [1]
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0:
            divs.extend([i, int(n / i)])
    divs.extend([n])
    return np.array(list(set(divs)))


def gen_gaussian_process(t: NPArray[float], kernel: str, params: dict) -> NPArray[float]:
    #: Match the kernel type
    match kernel:
        case 'periodic':
            #: Extract the parameters
            var, length_scale, period = params['var'], params['length_scale'], params['period']
            #: Define the kernel function
            k = lambda x_1, x_2: var * np.exp(-2 * np.sin(np.pi * np.abs(x_1 - x_2) / period) ** 2 / length_scale ** 2)
            #: Compute a meshgrid
            T_1, T_2 = np.meshgrid(t, t)
            #: Compute the covariance matrix
            K = k(T_1, T_2)
            #: Compute the Cholesky decomposition
            L = np.linalg.cholesky(K + 1e-6 * np.eye(len(t)))
            #: Generate the Gaussian process
            z = np.random.normal(size=len(t))
            gp = np.dot(L, z)
            #: Return the Gaussian process
            return gp
        case 'RBF':
            #: Extract the parameters
            var, length_scale = params['var'], params['length_scale']
            #: Define the kernel function
            k = lambda x_1, x_2: var * np.exp(-0.5 * (x_1 - x_2) ** 2 / length_scale ** 2)
            #: Compute a meshgrid
            T_1, T_2 = np.meshgrid(t, t)
            #: Compute the covariance matrix
            K = k(T_1, T_2)
            #: Compute the Cholesky decomposition
            L = np.linalg.cholesky(K + 1e-6 * np.eye(len(t)))
            #: Generate the Gaussian process
            z = np.random.normal(size=len(t))
            gp = np.dot(L, z)
            #: Return the Gaussian process
            return gp
        case 'white':
            #: Extract the parameters
            var = params['var']
            #: Generate the Gaussian process
            gp = np.random.normal(0, var, len(t))
            #: Return the Gaussian process
            return gp
        case _:
            raise ValueError(f"Unknown kernel type: {kernel}")