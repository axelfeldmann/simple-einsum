#!/usr/bin/env python3

from simple_einsum import einsum
import numpy as np

if __name__ == "__main__":

    A = np.random.rand(4, 5)
    B = np.random.rand(5, 6)

    expr = "ik,kj->ij"
    C_test = einsum(expr, A, B)
    C_ref = np.einsum(expr, A, B)

    print(f"{np.allclose(C_test, C_ref) = }")