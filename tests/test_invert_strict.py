import pytest
import numpy as np
from tvsclib import RealizationMixed, SeparationSVD, TransferOperator

def test_invert_strict():
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

    separation = SeparationSVD(0)
    T_1 = TransferOperator(matrix,dims_in,dims_out)
    R_1 = RealizationMixed(transferoperator=T_1,separation=separation)

    # Causal case
    R_strict = R_1.causal_system
    R_strict_inv = R_strict.inv()
    realization = R_strict_inv.realize()

    matrix_strict = R_strict.generate_transferoperator().matrix
    matrix_strict_inv = np.linalg.pinv(matrix_strict)
    rec_matrix_inv = realization.generate_transferoperator().matrix
    
    u = np.random.rand(matrix_strict_inv.shape[1],1)
    y = matrix_strict_inv @ u
    _, y_rec = R_strict_inv.compute(u)

    assert np.allclose(y,y_rec), "Invert matrix computation is wrong"
    assert np.allclose(matrix_strict_inv,rec_matrix_inv), "Invert matrix reconstruction is wrong"
    
    # Anticausal case
    R_strict = R_1.causal_system.transpose().realize()
    R_strict_inv = R_strict.inv()
    realization = R_strict_inv.realize()

    matrix_strict = R_strict.generate_transferoperator().matrix
    matrix_strict_inv = np.linalg.pinv(matrix_strict)
    rec_matrix_inv = realization.generate_transferoperator().matrix
    
    u = np.random.rand(matrix_strict_inv.shape[1],1)
    y = matrix_strict_inv @ u
    _, y_rec = R_strict_inv.compute(u)

    assert np.allclose(y,y_rec), "Invert matrix computation is wrong"
    assert np.allclose(matrix_strict_inv,rec_matrix_inv), "Invert matrix reconstruction is wrong"

    # Tall matrix
    matrix = np.random.rand(10,7)
    dims_in = [1, 3, 2, 1]
    dims_out = [2, 2, 4, 2]

    separation = SeparationSVD(0)
    T_1 = TransferOperator(matrix,dims_in,dims_out)
    R_1 = RealizationMixed(transferoperator=T_1,separation=separation)

    # Causal case
    R_strict = R_1.causal_system
    R_strict_inv = R_strict.inv()
    realization = R_strict_inv.realize()

    matrix_strict = R_strict.generate_transferoperator().matrix
    matrix_strict_inv = np.linalg.pinv(matrix_strict)
    rec_matrix_inv = realization.generate_transferoperator().matrix
    
    u = np.random.rand(matrix_strict_inv.shape[1],1)
    y = matrix_strict_inv @ u
    _, y_rec = R_strict_inv.compute(u)

    assert np.allclose(y,y_rec), "Invert matrix computation is wrong"
    assert np.allclose(matrix_strict_inv,rec_matrix_inv), "Invert matrix reconstruction is wrong"
    
    # Anticausal case
    R_strict = R_1.causal_system.transpose().realize()
    R_strict_inv = R_strict.inv()
    realization = R_strict_inv.realize()

    matrix_strict = R_strict.generate_transferoperator().matrix
    matrix_strict_inv = np.linalg.pinv(matrix_strict)
    rec_matrix_inv = realization.generate_transferoperator().matrix
    
    u = np.random.rand(matrix_strict_inv.shape[1],1)
    y = matrix_strict_inv @ u
    _, y_rec = R_strict_inv.compute(u)

    assert np.allclose(y,y_rec), "Invert matrix computation is wrong"
    assert np.allclose(matrix_strict_inv,rec_matrix_inv), "Invert matrix reconstruction is wrong"

    # Wide matrix
    matrix = np.random.rand(7,10)
    dims_out = [1, 3, 2, 1]
    dims_in = [2, 2, 4, 2]

    separation = SeparationSVD(0)
    T_1 = TransferOperator(matrix,dims_in,dims_out)
    R_1 = RealizationMixed(transferoperator=T_1,separation=separation)
    
    # Causal case
    R_strict = R_1.causal_system
    R_strict_inv = R_strict.inv()
    realization = R_strict_inv.realize()

    matrix_strict = R_strict.generate_transferoperator().matrix
    matrix_strict_inv = np.linalg.pinv(matrix_strict)
    rec_matrix_inv = realization.generate_transferoperator().matrix
    
    u = np.random.rand(matrix_strict_inv.shape[1],1)
    y = matrix_strict_inv @ u
    _, y_rec = R_strict_inv.compute(u)

    assert np.allclose(y,y_rec), "Invert matrix computation is wrong"
    assert np.allclose(matrix_strict_inv,rec_matrix_inv), "Invert matrix reconstruction is wrong"
    
    # Anticausal case
    R_strict = R_1.causal_system.transpose().realize()
    R_strict_inv = R_strict.inv()
    realization = R_strict_inv.realize()

    matrix_strict = R_strict.generate_transferoperator().matrix
    matrix_strict_inv = np.linalg.pinv(matrix_strict)
    rec_matrix_inv = realization.generate_transferoperator().matrix
    
    u = np.random.rand(matrix_strict_inv.shape[1],1)
    y = matrix_strict_inv @ u
    _, y_rec = R_strict_inv.compute(u)

    assert np.allclose(y,y_rec), "Invert matrix computation is wrong"
    assert np.allclose(matrix_strict_inv,rec_matrix_inv), "Invert matrix reconstruction is wrong"