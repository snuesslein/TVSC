import numpy as np
from tvsclib.strict_system import StrictSystem
from tvsclib.mixed_system import MixedSystem
from tvsclib.toeplitz_operator import ToeplitzOperator
from tvsclib.system_identification_svd import SystemIdentificationSVD
from tvsclib.expressions.const import Const
from tvsclib.expressions.add import Add
from tvsclib.expressions.multiply import Multiply
from tvsclib.expressions.invert import Invert
from tvsclib.expressions.transpose import Transpose
from tvsclib.expressions.negate import Negate

def testExpressions():
    matrix_A = np.random.rand(6,6)
    matrix_B = np.random.rand(6,6)
    matrix_C = np.random.rand(6,6)
    dims_in =  [2, 1, 2, 1]
    dims_out = [2, 1, 2, 1]
    T_A = ToeplitzOperator(matrix_A, dims_in, dims_out)
    S_A = SystemIdentificationSVD(T_A)
    T_B = ToeplitzOperator(matrix_B, dims_in, dims_out)
    S_B = SystemIdentificationSVD(T_B)
    T_C = ToeplitzOperator(matrix_C, dims_in, dims_out)
    S_C = SystemIdentificationSVD(T_C)

    system_A = MixedSystem(S_A)
    system_B = MixedSystem(S_B)
    system_C = MixedSystem(S_C)

    A = Const(system_A.causal_system, "A")
    B = Const(system_B.causal_system, "B")
    C = Const(system_C.causal_system, "C")

    u = np.random.rand(6,1)

    add = Add(A, B)
    inv = Invert(add)
    mul = Multiply(inv, C)
    trp = Transpose(mul)
    comp_1 = trp.compile()
    mat_1_ref = (np.linalg.inv(A.operand.to_matrix() + B.operand.to_matrix()) @ C.operand.to_matrix()).transpose()
    mat_1_rec = comp_1.realize().to_matrix()
    y_1 = comp_1.compute(u)
    y_1_ref = mat_1_ref @ u
    assert np.allclose(y_1,y_1_ref), "Computation wrong"
    assert np.allclose(mat_1_ref,mat_1_rec), "Reconstruction wrong"

    inv_A = Invert(A)
    inv_B = Invert(B)
    mul = Multiply(inv_A, inv_B)
    trp_1 = Transpose(mul)
    inv_trp = Invert(trp_1)
    trp_2 = Transpose(inv_trp)
    comp_2 = trp_2.compile()
    mat_2_ref = B.operand.to_matrix() @ A.operand.to_matrix()
    mat_2_rec = comp_2.realize().to_matrix()
    y_2 = comp_2.compute(u)
    y_2_ref = mat_2_ref @ u
    assert np.allclose(y_2,y_2_ref), "Computation wrong"
    assert np.allclose(mat_2_ref,mat_2_rec), "Reconstruction wrong"


    trp_A = Transpose(A)
    trp_B = Transpose(B)
    mul = Multiply(trp_A, trp_B)
    inv_1 = Invert(mul)
    trp_inv = Transpose(inv_1)
    inv_2 = Invert(trp_inv)
    comp_3 = inv_2.compile()
    mat_3_ref = B.operand.to_matrix() @ A.operand.to_matrix()
    mat_3_rec = comp_3.realize().to_matrix()
    y_3 = comp_3.compute(u)
    y_3_ref = mat_3_ref @ u
    assert np.allclose(y_3,y_3_ref), "Computation wrong"
    assert np.allclose(mat_3_ref,mat_3_rec), "Reconstruction wrong"


    inv_A = Invert(A)
    inv_B = Invert(B)
    mul = Multiply(inv_A, inv_B)
    trp_1 = Transpose(mul)
    inv_trp = Invert(trp_1)
    trp_2 = Transpose(inv_trp)
    trp_3 = Transpose(trp_2)
    comp_4 = trp_3.compile()
    mat_4_ref = A.operand.to_matrix().transpose() @ B.operand.to_matrix().transpose()
    mat_4_rec = comp_4.realize().to_matrix()
    y_4 = comp_4.compute(u)
    y_4_ref = mat_4_ref @ u
    assert np.allclose(y_4,y_4_ref), "Computation wrong"
    assert np.allclose(mat_4_ref,mat_4_rec), "Reconstruction wrong"


    trp_A = Transpose(A)
    trp_B = Transpose(B)
    mul = Multiply(trp_A, trp_B)
    inv_1 = Invert(mul)
    trp_inv = Transpose(inv_1)
    inv_2 = Invert(trp_inv)
    inv_3 = Invert(inv_2)
    comp_5 = inv_3.compile()
    mat_5_ref = np.linalg.inv(A.operand.to_matrix()) @ np.linalg.inv(B.operand.to_matrix())
    mat_5_rec = comp_5.realize().to_matrix()
    y_5 = comp_5.compute(u)
    y_5_ref = mat_5_ref @ u
    assert np.allclose(y_5,y_5_ref), "Computation wrong"
    assert np.allclose(mat_5_ref,mat_5_rec), "Reconstruction wrong"