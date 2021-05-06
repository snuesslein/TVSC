import pytest
import numpy as np
from tvsclib import RealizationStrict, SeparationSVD, TransferOperator

def test_multiply_strict():
    matrix_lhs = np.array([
        [5,     4,     0,     0,     0,     0],
        [2,     3,     2,     0,     0,     0],
        [6,     3,     5,     0,     0,     0],
        [3,     5,     5,     5,     3,     0],
        [2,     4,     3,     6,     1,     2],
        [2,     4,     4,     1,     5,     4]
    ])
    dims_in_lhs = [2, 1, 2, 1]
    dims_out_lhs = [1, 2, 1, 2]

    matrix_rhs = np.array([
        [5,     4,     0,     0,     0,     0],
        [2,     3,     0,     0,     0,     0],
        [6,     3,     5,     2,     0,     0],
        [3,     5,     5,     5,     3,     0],
        [2,     4,     3,     6,     1,     0],
        [2,     4,     4,     1,     5,     4]
    ])
    dims_in_rhs = [2, 2, 1, 1]
    dims_out_rhs = [2, 1, 2, 1]

    T_lhs = TransferOperator(matrix_lhs,dims_in_lhs,dims_out_lhs)
    T_rhs = TransferOperator(matrix_rhs,dims_in_rhs,dims_out_rhs)
    separation = SeparationSVD(0)
    
    R_lhs = RealizationStrict(causal=True,transferoperator=T_lhs,separation=separation)
    R_rhs = RealizationStrict(causal=True,transferoperator=T_rhs,separation=separation)

    u = np.array([1,2,3,4,5,6]).reshape((6,1))

    y_ref_lhs_rhs = (matrix_lhs @ matrix_rhs) @ u
    mul_lhs_rhs = R_lhs.mul(R_rhs)
    x_mul_lhs_rhs,y_mul_lhs_rhs = mul_lhs_rhs.compute(u)
    R_mul_lhs_rhs = mul_lhs_rhs.realize()
    x_R_mul_lhs_rhs,y_R_mul_lhs_rhs = R_mul_lhs_rhs.compute(u)
    matrix_rec_lhs_rhs = R_mul_lhs_rhs.generate_transferoperator().matrix

    assert np.allclose(y_ref_lhs_rhs,y_mul_lhs_rhs), "Multiplication computation is wrong"
    assert np.allclose(y_ref_lhs_rhs,y_R_mul_lhs_rhs), "Multiplication realization is wrong"
    assert np.allclose(matrix_lhs@matrix_rhs,matrix_rec_lhs_rhs), "Multiplication reconstruction is wrong"

    y_ref_lhs_lhs = (matrix_lhs @ matrix_lhs) @ u
    mul_lhs_lhs = R_lhs.mul(R_lhs)
    x_mul_lhs_lhs,y_mul_lhs_lhs = mul_lhs_lhs.compute(u)
    assert np.allclose(y_ref_lhs_lhs,y_mul_lhs_lhs), "Multiplication computation is wrong"
    with pytest.raises(AttributeError):
        R_mul_lhs_lhs = mul_lhs_lhs.realize()
