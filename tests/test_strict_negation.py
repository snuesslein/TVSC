import numpy as np
from tvsclib.mixed_system import MixedSystem
from tvsclib.toeplitz_operator import ToeplitzOperator
from tvsclib.system_identification_svd import SystemIdentificationSVD
from tvsclib.expressions.const import Const
from tvsclib.expressions.negate import Negate

def testStrictNegation():
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

    system = MixedSystem(S).causal_system
    y = -system.to_matrix()@u
    c = Const(system)
    n = Negate(c)

    y_rec = n.compute(u)
    matrix_rec = n.realize().to_matrix()
    matrix_ref = -system.to_matrix()

    assert np.allclose(y, y_rec), "Negation computation is wrong"
    assert np.allclose(matrix_ref, matrix_rec), "Negation matrix reconstruction is wrong"
