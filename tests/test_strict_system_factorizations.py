import numpy as np
from tvsclib.canonical_form import CanonicalForm
from tvsclib.mixed_system import MixedSystem
from tvsclib.toeplitz_operator import ToeplitzOperator
from tvsclib.system_identification_svd import SystemIdentificationSVD
from tvsclib.transformations.reduction import Reduction
from tvsclib.expressions.multiply import Multiply
from tvsclib.expressions.invert import Invert
from tvsclib.expressions.const import Const

def testStrictSystemFactorization():
    # Square Matrix
    matrix = np.random.rand(10,10)
    dims_in =  [2, 4, 1, 3]
    dims_out = [2, 1, 4, 3]
    T = ToeplitzOperator(matrix, dims_in, dims_out)
    S = SystemIdentificationSVD(T,CanonicalForm.BALANCED)

    system_balanced = MixedSystem(S)

    system = system_balanced.causal_system
    T_o,V = system.outer_inner_factorization()
    T_o_inv = T_o.arrow_reversal()
    T_o_inv_rec = T_o_inv.to_matrix()
    system_rec = Multiply(Const(T_o), Const(V)).realize()
    matrix_rec = system_rec.to_matrix()
    I_system = Multiply(Const(V), Const(V.transpose())).realize()
    I_rec = I_system.to_matrix()

    T_o_inv_ref = np.linalg.inv(T_o.to_matrix())
    I_ref = np.eye(I_rec.shape[0])
    matrix_ref = system.to_matrix()

    assert np.allclose(I_ref, I_rec), "V is not unitary"
    assert np.allclose(T_o_inv_ref, T_o_inv_rec), "Arrow reversal is wrong"
    assert np.allclose(matrix_ref, matrix_rec), "Outer inner product is wrong"

    
    system = system_balanced.causal_system.transpose()
    T_o,V = system.outer_inner_factorization()
    T_o_inv = T_o.arrow_reversal()
    T_o_inv_rec = T_o_inv.to_matrix()
    system_rec = Multiply(Const(T_o), Const(V)).realize()
    matrix_rec = system_rec.to_matrix()
    I_system = Multiply(Const(V), Const(V.transpose())).realize()
    I_rec = I_system.to_matrix()

    T_o_inv_ref = np.linalg.inv(T_o.to_matrix())
    I_ref = np.eye(I_rec.shape[0])
    matrix_ref = system.to_matrix()

    assert np.allclose(I_ref, I_rec), "V is not unitary"
    assert np.allclose(T_o_inv_ref, T_o_inv_rec), "Arrow reversal is wrong"
    assert np.allclose(matrix_ref, matrix_rec), "Outer inner product is wrong"

    # Tall Matrix
    matrix = np.random.rand(20,10)
    dims_in =  [2, 4, 1, 3]
    dims_out = [4, 2, 8, 6]
    T = ToeplitzOperator(matrix, dims_in, dims_out)
    S = SystemIdentificationSVD(T,CanonicalForm.BALANCED)

    system_balanced = MixedSystem(S)

    system = system_balanced.causal_system
    T_o,V = system.outer_inner_factorization()
    T_o_inv = T_o.arrow_reversal()
    T_o_inv_rec = T_o_inv.to_matrix()
    system_rec = Multiply(Const(T_o), Const(V)).realize()
    matrix_rec = system_rec.to_matrix()
    I_system = Multiply(Const(V), Const(V.transpose())).realize()
    I_rec = I_system.to_matrix()

    I_ref = np.eye(I_rec.shape[0])
    matrix_ref = system.to_matrix()

    I_T_o = T_o_inv_rec @ T_o.to_matrix()
    I_T_o_ref = np.eye(I_T_o.shape[0])

    assert np.allclose(I_ref, I_rec), "V is not unitary"
    assert np.allclose(I_T_o_ref, I_T_o), "Arrow reversal is wrong"
    assert np.allclose(matrix_ref, matrix_rec), "Outer inner product is wrong"

    
    system = system_balanced.causal_system.transpose()
    T_o,V = system.outer_inner_factorization()
    T_o_inv = T_o.arrow_reversal()
    T_o_inv_rec = T_o_inv.to_matrix()
    system_rec = Multiply(Const(T_o), Const(V)).realize()
    matrix_rec = system_rec.to_matrix()
    I_system = Multiply(Const(V), Const(V.transpose())).realize()
    I_rec = I_system.to_matrix()

    I_ref = np.eye(I_rec.shape[0])
    matrix_ref = system.to_matrix()

    I_T_o = T_o.to_matrix() @ T_o_inv_rec
    I_T_o_ref = np.eye(I_T_o.shape[0])

    assert np.allclose(I_ref, I_rec), "V is not unitary"
    assert np.allclose(I_T_o_ref, I_T_o), "Arrow reversal is wrong"
    assert np.allclose(matrix_ref, matrix_rec), "Outer inner product is wrong"
    
    # Wide Matrix
    matrix = np.random.rand(10,20)
    dims_in =  [4, 2, 8, 6]
    dims_out = [2, 4, 1, 3]
    T = ToeplitzOperator(matrix, dims_in, dims_out)
    S = SystemIdentificationSVD(T,CanonicalForm.BALANCED)

    system_balanced = MixedSystem(S)

    system = system_balanced.causal_system
    T_o,V = system.outer_inner_factorization()
    T_o_inv = T_o.arrow_reversal()
    T_o_inv_rec = T_o_inv.to_matrix()
    system_rec = Multiply(Const(T_o), Const(V)).realize()
    matrix_rec = system_rec.to_matrix()
    I_system = Multiply(Const(V), Const(V.transpose())).realize()
    I_rec = I_system.to_matrix()

    I_ref = np.eye(I_rec.shape[0])
    matrix_ref = system.to_matrix()

    I_T_o = T_o.to_matrix() @ T_o_inv_rec
    I_T_o_ref = np.eye(I_T_o.shape[0])

    assert np.allclose(I_ref, I_rec), "V is not unitary"
    assert np.allclose(I_T_o_ref, I_T_o), "Arrow reversal is wrong"
    assert np.allclose(matrix_ref, matrix_rec), "Outer inner product is wrong"

    
    system = system_balanced.causal_system.transpose()
    T_o,V = system.outer_inner_factorization()
    T_o_inv = T_o.arrow_reversal()
    T_o_inv_rec = T_o_inv.to_matrix()
    system_rec = Multiply(Const(T_o), Const(V)).realize()
    matrix_rec = system_rec.to_matrix()
    I_system = Multiply(Const(V), Const(V.transpose())).realize()
    I_rec = I_system.to_matrix()

    I_ref = np.eye(I_rec.shape[0])
    matrix_ref = system.to_matrix()

    I_T_o = T_o_inv_rec @ T_o.to_matrix()
    I_T_o_ref = np.eye(I_T_o.shape[0])

    assert np.allclose(I_ref, I_rec), "V is not unitary"
    assert np.allclose(I_T_o_ref, I_T_o), "Arrow reversal is wrong"
    assert np.allclose(matrix_ref, matrix_rec), "Outer inner product is wrong"