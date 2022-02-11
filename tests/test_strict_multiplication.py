import numpy as np
from tvsclib.mixed_system import MixedSystem
from tvsclib.toeplitz_operator import ToeplitzOperator
from tvsclib.system_identification_svd import SystemIdentificationSVD
from tvsclib.expressions.const import Const
from tvsclib.expressions.multiply import Multiply

def testStrictMultiplication():
    matrix = np.array([
        [5,     4,     6,     1,     4,     2],
        [2,     3,     2,     1,     3,     4],
        [6,     3,     5,     4,     1,     1],
        [3,     5,     5,     5,     3,     4],
        [2,     4,     3,     6,     1,     2],
        [2,     4,     4,     1,     5,     4]
    ])
    dims_in =  [2, 1, 2, 1]
    dims_out = [2, 1, 2, 1]
    T = ToeplitzOperator(matrix, dims_in, dims_out)
    S = SystemIdentificationSVD(T)

    u = np.array([1,2,3,4,5,6]).reshape((6,1))

    system = MixedSystem(S)
    matrix_ref = system.causal_system.to_matrix() @ system.causal_system.to_matrix()
    y_ref = matrix_ref @ u
    val_1 = Const(system.causal_system)
    val_2 = Const(system.causal_system)
    mul = Multiply(val_1, val_2)

    system_mul = mul.realize()
    matrix_rec = system_mul.to_matrix()
    x,y = system_mul.compute(u)


    assert np.allclose(y_ref, y), "Multiplication computation is wrong"
    assert np.allclose(matrix_ref, matrix_rec), "Multiplication matrix reconstruction is wrong"
