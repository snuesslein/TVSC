import pytest
import numpy as np
from tvsclib import (
    RealizationStrict,
    RealizationMixed,
    SeparationSVD,
    TransferOperator)

def test_add_mixed():
    # Adding causal and anticausal system
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

    separation = SeparationSVD(0)
    T_1 = TransferOperator(matrix,dims_in,dims_out)
    R_1 = RealizationStrict(causal=True,transferoperator=T_1,separation=separation)
    T_2 = TransferOperator(matrix,dims_out,dims_in)
    R_2 = RealizationStrict(causal=True,transferoperator=T_2,separation=separation)
    
    matrix_1 = R_1.generate_transferoperator().matrix
    matrix_2 = R_2.generate_transferoperator().matrix

    u = np.array([1,2,3,4,5,6]).reshape((6,1))

    matrix_ref = matrix_1 + matrix_2.transpose()
    y_ref = matrix_ref @ u
    add_result = R_1.add(R_2.transpose())
    x_add,y_add = add_result.compute(u)
    R_add_result = add_result.realize()
    x_R_add,y_R_add = R_add_result.compute(u)
    matrix_rec = R_add_result.generate_transferoperator().matrix
    
    assert np.allclose(y_ref,y_add), "Addition computation is wrong"
    assert np.allclose(y_ref,y_R_add), "Addition realization is wrong"
    assert np.allclose(matrix_ref,matrix_rec), "Multiplication reconstruction is wrong"

    # Adding mixed systems
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

    T = TransferOperator(matrix,dims_in,dims_out)
    separation = SeparationSVD(0)

    R_1 = RealizationMixed(separation=separation,transferoperator=T)
    R_2 = RealizationMixed(separation=separation,transferoperator=T)
    
    y_ref = (matrix + matrix)@u
    add_result = R_1.add(R_2)
    x_add,y_add = add_result.compute(u)

    assert np.allclose(y_ref,y_add), "Addition computation is wrong"
