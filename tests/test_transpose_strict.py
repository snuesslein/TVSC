import pytest
import numpy as np
from tvsclib import RealizationStrict, SeparationSVD, TransferOperator

def test_transpose_strict():
    matrix = np.array([
        [5,     4,     0,     0,     0,     0],
        [2,     3,     2,     0,     0,     0],
        [6,     3,     5,     0,     0,     0],
        [3,     5,     5,     5,     3,     0],
        [2,     4,     3,     6,     1,     2],
        [2,     4,     4,     1,     5,     4]
    ])
    dims_in = [2, 1, 2, 1]
    dims_out = [1, 2, 1, 2]

    T = TransferOperator(matrix,dims_in,dims_out)
    separation = SeparationSVD(0)
    R = RealizationStrict(causal=True,transferoperator=T,separation=separation)
    
    u = np.array([1,2,3,4,5,6]).reshape((6,1))

    y_ref = matrix.transpose() @ u
    trans = R.transpose()
    x_trans,y_trans = trans.compute(u)
    R_trans = trans.realize()
    x_R_trans,y_R_trans = R_trans.compute(u)
    matrix_rec = R_trans.generate_transferoperator().matrix

    assert np.allclose(y_ref,y_trans), "Transposition computation is wrong"
    assert np.allclose(y_ref,y_R_trans), "Transposition realization is wrong"
    assert np.allclose(matrix.transpose(),matrix_rec), "Transposition reconstruction is wrong"
    assert R_trans.causal is not R.causal, "Transposition doesnt change causality"
