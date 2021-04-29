from tvsclib import RealizationStrict, SeparationSVD, TransferOperator
import numpy as np

def test_realization_strict():
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

    R_causal = RealizationStrict(causal=True,transferoperator=T,separation=separation)
    T_rec_causal = R_causal.generate_transferoperator()

    R_anticausal = RealizationStrict(causal=False,transferoperator=T,separation=separation)
    T_rec_anticausal = R_anticausal.generate_transferoperator()

    (x_causal,y_causal) = R_causal.compute(u)
    (x_anticausal,y_anticausal) = R_anticausal.compute(u)

    matrix_rec = T_rec_causal.matrix + T_rec_anticausal.matrix
    y_rec = y_causal + y_anticausal

    assert np.allclose(matrix_rec,matrix), "Reconstructed matrix is different"
    assert np.allclose(y_rec,y), "Applying a vector leads to wrong result"
    assert np.all(T_rec_causal.dims_in == dims_in), "Input dimensions are wrong"
    assert np.all(T_rec_causal.dims_out == dims_out), "Output dimensions are wrong"
    assert np.all(T_rec_anticausal.dims_in == dims_in), "Input dimensions are wrong"
    assert np.all(T_rec_anticausal.dims_out == dims_out), "Output dimensions are wrong"

