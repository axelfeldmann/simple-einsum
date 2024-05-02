import pytest

from simple_einsum import einsum
import numpy as np
from itertools import product

matmul_sizes = product([ 1, 5, 8, 10 ], repeat=3)
@pytest.mark.parametrize("I, J, K", matmul_sizes)
def test_matmul(I, J, K):
    a = np.random.rand(I, J)
    b = np.random.rand(J, K)
    expr = "ij,jk->ik"
    assert np.allclose(einsum(expr, a, b), np.einsum(expr, a, b))

outerprod_sizes = product([ 1, 5, 8, 10 ], repeat=2)
@pytest.mark.parametrize("I, J", outerprod_sizes)
def test_outerprod(I, J):
    a = np.random.rand(I)
    b = np.random.rand(J)
    expr = "i,j->ij"
    assert np.allclose(einsum(expr, a, b), np.einsum(expr, a, b))

batched_matmul_sizes = product([ 1, 5, 8, 10 ], repeat=4)
@pytest.mark.parametrize("B, I, J, K", batched_matmul_sizes)
def test_batched_matmul(B, I, J, K):
    a = np.random.rand(B, I, J)
    b = np.random.rand(B, J, K)
    expr = "bij,bjk->bik"
    assert np.allclose(einsum(expr, a, b), np.einsum(expr, a, b))

bilinear_sizes = product([ 1, 5, 8, 10 ], repeat=4)
@pytest.mark.parametrize("I, J, K, L", bilinear_sizes)
def test_bilinear(I, J, K, L):
    a = np.random.rand(I, J)
    b = np.random.rand(L, J, K)
    c = np.random.rand(I, K)
    expr = "ij,ljk,ik->il"
    assert np.allclose(einsum(expr, a, b, c), np.einsum(expr, a, b, c))

tensor_prod_sizes = product([ 1, 5, 8, 10 ], repeat=4)
@pytest.mark.parametrize("I, J, K, L", tensor_prod_sizes)
def test_tensor_prod(I, J, K, L):
    a = np.random.rand(I, J, K)
    b = np.random.rand(J, K, L)
    expr = "ijk,jkl->il"
    assert np.allclose(einsum(expr, a, b), np.einsum(expr, a, b))
