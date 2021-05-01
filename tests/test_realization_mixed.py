from tvsclib import RealizationMixed, SeparationSVD, TransferOperator
import numpy as np

def test_realization_mixed():
    matrix = np.array([
        [5,     4,     6,     1,     4,     2],
        [2,     3,     2,     1,     3,     4],
        [6,     3,     5,     4,     1,     1],
        [3,     5,     5,     5,     3,     4],
        [2,     4,     3,     6,     1,     2],
        [2,     4,     4,     1,     5,     4]
    ])
    u = np.array([1,2,3,4,5,6]).reshape((6,1))
    dims_in =  [2, 1, 2, 1]
    dims_out = [1, 2, 1, 2]

    y = matrix@u

    T = TransferOperator(matrix,dims_in,dims_out)
    separation = SeparationSVD(0)

    R_mixed = RealizationMixed(separation=separation,transferoperator=T)
    T_mixed = R_mixed.generate_transferoperator()
    (x_mixed,y_mixed) = R_mixed.compute(u)

    assert np.allclose(T_mixed.matrix,matrix), "Reconstructed matrix is different"
    assert np.allclose(y_mixed,y),  "Applying a vector leads to wrong result"