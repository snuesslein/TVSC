import numpy as np
from tvsclib.mixed_system import MixedSystem
from tvsclib.toeplitz_operator import ToeplitzOperator
from tvsclib.system_identification_svd import SystemIdentificationSVD
from tvsclib.expressions.const import Const
from tvsclib.expressions.multiply import Multiply
from tvsclib.expressions.transpose import Transpose

def testMixedMultiplication():
    matrix = np.array([
        [5,     4,     6,     1,     4,     2],
        [2,     3,     2,     1,     3,     4],
        [6,     3,     5,     4,     1,     1],
        [3,     5,     5,     5,     3,     4],
        [2,     4,     3,     6,     1,     2],
        [2,     4,     4,     1,     5,     4]
    ])
    dims_in =  [2, 1, 2, 1]
    dims_out = [1, 2, 1, 2]
    T = ToeplitzOperator(matrix, dims_in, dims_out)
    S = SystemIdentificationSVD(T)

    u = np.array([1,2,3,4,5,6]).reshape((6,1))

    system_a = MixedSystem(S)
    system_b = MixedSystem(S)
    mul = Multiply(Const(system_a), Transpose(Const(system_b)))
    mul_realization = mul.realize()

    matrix_ref = matrix @ matrix.transpose()
    matrix_rec = mul_realization.to_matrix()
    
    y_ref = matrix_ref @ u
    y_rec = mul.compute(u)

    assert np.allclose(y_ref, y_rec), "Multiplication computation is wrong"
    assert np.allclose(matrix_ref, matrix_rec), "Multiplication matrix reconstruction is wrong"
