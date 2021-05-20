import pytest
import numpy as np
from tvsclib import RealizationMixed, SeparationSVD, TransferOperator

def test_invert_mixed():
    # Square matrix
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

    y_ref = np.linalg.inv(matrix) @ u    
    inv = R_1.inv()
    R_inv = inv.realize()
    T_inv = R_inv.generate_transferoperator()
    x_inv,y_inv = inv.compute(u)
    x_R_inv,y_R_inv = R_inv.compute(u)

    assert np.allclose(y_ref,y_inv), "Invert square matrix computation is wrong"
    assert np.allclose(y_ref,y_R_inv), "Invert square matrix realization is wrong"
    assert np.allclose(np.linalg.inv(matrix),T_inv.matrix), "Invert square matrix reconstruction is wrong"

    # Tall matrix
    matrix = np.array([
        [5,     4,     8,     5],
        [2,     3,     2,     3],
        [6,     3,     5,     2],
        [3,     5,     5,     5],
        [2,     4,     3,     6],
        [2,     4,     4,     1]
    ])
    dims_in = [1, 2, 1]
    dims_out = [1, 2, 3]

    u = np.array([1,2,3,4,5,6]).reshape((6,1))

    separation = SeparationSVD(0)
    T_1 = TransferOperator(matrix,dims_in,dims_out)
    R_1 = RealizationMixed(transferoperator=T_1,separation=separation)

    y_ref = np.linalg.pinv(matrix) @ u    
    inv = R_1.inv()
    R_inv = inv.realize()
    T_inv = R_inv.generate_transferoperator()
    x_inv,y_inv = inv.compute(u)
    x_R_inv,y_R_inv = R_inv.compute(u)

    assert np.allclose(y_ref,y_inv), "Invert tall matrix computation is wrong"
    assert np.allclose(y_ref,y_R_inv), "Invert tall matrix realization is wrong"
    assert np.allclose(np.linalg.pinv(matrix),T_inv.matrix), "Invert tall matrix reconstruction is wrong"

    # Wide matrix
    matrix = np.array([
        [5,     4,     8,     5],
        [2,     3,     2,     3],
        [6,     3,     5,     2],
        [3,     5,     5,     5],
        [2,     4,     3,     6],
        [2,     4,     4,     1]
    ]).transpose()
    dims_in = [1, 2, 3]
    dims_out = [1, 2, 1]

    u = np.array([1,2,3,4]).reshape((4,1))

    separation = SeparationSVD(0)
    T_1 = TransferOperator(matrix,dims_in,dims_out)
    R_1 = RealizationMixed(transferoperator=T_1,separation=separation)

    y_ref = np.linalg.pinv(matrix) @ u    
    inv = R_1.inv()
    R_inv = inv.realize()
    T_inv = R_inv.generate_transferoperator()
    x_inv,y_inv = inv.compute(u)
    x_R_inv,y_R_inv = R_inv.compute(u)

    assert np.allclose(y_ref,y_inv), "Invert wide matrix computation is wrong"
    assert np.allclose(y_ref,y_R_inv), "Invert wide matrix realization is wrong"
    assert np.allclose(np.linalg.pinv(matrix),T_inv.matrix), "Invert wide matrix reconstruction is wrong"

