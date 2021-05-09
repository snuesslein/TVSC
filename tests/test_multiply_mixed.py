import pytest
import numpy as np
from tvsclib import RealizationMixed, SeparationSVD, TransferOperator

def test_multiply_mixed():
    matrix = np.array([
        [5,     4,     8,     5,     1,     4],
        [2,     3,     2,     3,     2,     5],
        [6,     3,     5,     2,     4,     9],
        [3,     5,     5,     5,     3,     6],
        [2,     4,     3,     6,     1,     2],
        [2,     4,     4,     1,     5,     4]
    ])
    dims_in = [2, 1, 2, 1]
    dims_out = [1, 2, 1, 2]

    u = np.array([1,2,3,4,5,6]).reshape((6,1))

    separation = SeparationSVD(0)
    T_1 = TransferOperator(matrix,dims_in,dims_out)
    R_1 = RealizationMixed(transferoperator=T_1,separation=separation)
    T_2 = TransferOperator(matrix,dims_out,dims_in)
    R_2 = RealizationMixed(transferoperator=T_2,separation=separation)
    
    muli = R_1.mul(R_2).realize()
    R_muli = muli.realize()
    T_muli = R_muli.generate_transferoperator()
    matrix_muli = T_muli.matrix

    y_ref = (matrix@matrix)@u
    x_muli,y_muli = muli.compute(u)
    x_R_muli,y_R_muli = R_muli.compute(u)

    assert np.allclose(y_ref,y_muli), "Multiplication computation is wrong"
    assert np.allclose(y_ref,y_R_muli), "Multiplication realization is wrong"
    assert np.allclose(matrix@matrix,matrix_muli), "Multiplication reconstruction is wrong"