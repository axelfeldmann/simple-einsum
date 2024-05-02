import numpy as np
from itertools import chain, product
from functools import reduce

np.random.seed(0)

class NamedDimTensor:
    def __init__(self, tensor, *args):
        self.tensor = tensor
        self.dims = { dim: i for i, dim in enumerate(args) }

    def get_slice(self, **kwargs):
        dims, indices = zip(*[ (self.dims[dim], idx) for dim, idx in kwargs.items() if dim in self.dims  ])
        slicers = [slice(None)] * self.tensor.ndim
        for dim, index in zip(dims, indices):
            slicers[dim] = index
        return self.tensor[tuple(slicers)]

    def get_dim_sizes(self):
        return { dim: self.tensor.shape[i] for dim, i in self.dims.items() }

def two_input_einsum(A, B, final_dims, dim_sizes):

    contracted_dims = set(A.dims.keys()) & set(B.dims.keys()) - set(final_dims)
    output_dims = (set(A.dims.keys()) | set(B.dims.keys())) - contracted_dims

    # sort the dims so that dims that are final appear in the correct order
    # and the dims that will be contracted during future two_input_einsums
    # are at the end in some arbitrary order
    output_dims = sorted(output_dims, key=lambda x: final_dims.index(x) if x in final_dims else -1)
    output = np.zeros([ dim_sizes[dim] for dim in output_dims ])

    # iterate over all the possible positions in the output tensor and compute the
    # contracted product
    for coords in product(*[ range(dim_sizes[dim]) for dim in output_dims ]):
        coord_dict = dict(zip(output_dims, coords))
        A_slice = A.get_slice(**coord_dict)
        B_slice = B.get_slice(**coord_dict)
        output[coords] = np.sum(A_slice * B_slice)
    return NamedDimTensor(output, *output_dims)

def einsum(expr, *args):

    lhs, output_dims = expr.split("->")
    lhs_idxs = [ list(idxs) for idxs in lhs.split(",") ]

    assert len(lhs_idxs) == len(args)
    assert len(lhs_idxs) >= 2, "one input einsums not supported for now"

    inputs = [ NamedDimTensor(arg, *idxs) for idxs, arg in zip(lhs_idxs, args) ]

    # Make sure that input sizes are consistent
    dim_sizes = {}
    for tensor in inputs:
        input_sizes = tensor.get_dim_sizes()
        for dim, size in input_sizes.items():
            if dim in dim_sizes:
                assert dim_sizes[dim] == size
            else:
                dim_sizes[dim] = size
    
    # Left to right evaluation. This is not optimal at all, but whatever
    cur = two_input_einsum(inputs[0], inputs[1], output_dims, dim_sizes)
    for i in range(2, len(inputs)):
        cur = two_input_einsum(cur, inputs[i], output_dims, dim_sizes)

    return cur.tensor