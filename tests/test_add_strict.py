import pytest
import numpy as np
from tvsclib import RealizationStrict, SeparationSVD, TransferOperator

def test_add_strict():
    matrix = np.array([
        [5,     4,     6,     1,     4,     2],
        [2,     3,     2,     1,     3,     4],
        [6,     3,     5,     4,     1,     1],
        [3,     5,     5,     5,     3,     4],
        [2,     4,     3,     6,     1,     2],
        [2,     4,     4,     1,     5,     4]
    ])
    matrix_causal = np.array([
        [5,     4,     0,     0,     0,     0],
        [2,     3,     2,     0,     0,     0],
        [6,     3,     5,     0,     0,     0],
        [3,     5,     5,     5,     3,     0],
        [2,     4,     3,     6,     1,     2],
        [2,     4,     4,     1,     5,     4]
    ])
    matrix_anticausal = matrix - matrix_causal
    u = np.array([1,2,3,4,5,6]).reshape((6,1))
    dims_in =  [2, 1, 2, 1]
    dims_out = [1, 2, 1, 2]

    T = TransferOperator(matrix,dims_in,dims_out)
    separation = SeparationSVD(0)

    R_causal_1 = RealizationStrict(causal=True,transferoperator=T,separation=separation)
    R_causal_2 = RealizationStrict(causal=True,transferoperator=T,separation=separation)
    R_anticausal_1 = RealizationStrict(causal=False,transferoperator=T,separation=separation)
    R_anticausal_2 = RealizationStrict(causal=False,transferoperator=T,separation=separation)

    R_causal_add = R_causal_1.add(R_causal_2)
    R_anticausal_add = R_anticausal_1.add(R_anticausal_2)

    (x_causal,y_causal) = R_causal_add.compute(u)
    (x_anticausal,y_anticausal) = R_anticausal_add.compute(u)

    T_rec_causal = R_causal_add.realize().generate_transferoperator()
    T_rec_anticausal = R_anticausal_add.realize().generate_transferoperator()

    y_ref_causal = (matrix_causal + matrix_causal)@u
    y_ref_anticausal = (matrix_anticausal + matrix_anticausal)@u

    assert np.allclose(T_rec_causal.matrix,matrix_causal + matrix_causal), "Reconstructed matrix is different"
    assert np.allclose(T_rec_anticausal.matrix,matrix_anticausal + matrix_anticausal), "Reconstructed matrix is different"
    assert np.allclose(y_causal,y_ref_causal), "Applying a vector leads to wrong result"
    assert np.allclose(y_anticausal,y_ref_anticausal), "Applying a vector leads to wrong result"
    assert np.all(T_rec_causal.dims_in == dims_in), "Input dimensions are wrong"
    assert np.all(T_rec_causal.dims_out == dims_out), "Output dimensions are wrong"
    assert np.all(T_rec_anticausal.dims_in == dims_in), "Input dimensions are wrong"
    assert np.all(T_rec_anticausal.dims_out == dims_out), "Output dimensions are wrong"

    # Checking exceptions
    dims_in_wrong =  [1, 1, 2, 2]
    dims_out_wrong = [2, 2, 1, 1]
    T_wrong = TransferOperator(matrix,dims_in_wrong,dims_out_wrong)
    R_causal_wrong = RealizationStrict(causal=True,transferoperator=T_wrong,separation=separation)
    with pytest.raises(AttributeError):
        R_add_wrong = R_causal_1.add(R_causal_wrong).realize()

