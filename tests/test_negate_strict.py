import pytest
import numpy as np
from tvsclib import RealizationStrict, SeparationSVD, TransferOperator

def test_negate_strict():
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

    y_ref = -matrix @ u
    neg = R.neg()
    x_neg,y_neg = neg.compute(u)
    R_neg = neg.realize()
    x_R_neg,y_R_neg = R_neg.compute(u)
    matrix_rec = R_neg.generate_transferoperator().matrix

    assert np.allclose(y_ref,y_neg), "Negation computation is wrong"
    assert np.allclose(y_ref,y_R_neg), "Negation realization is wrong"
    assert np.allclose(-matrix,matrix_rec), "Negation reconstruction is wrong"
