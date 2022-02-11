import numpy as np
from tvsclib.mixed_system import MixedSystem
from tvsclib.toeplitz_operator import ToeplitzOperator
from tvsclib.system_identification_svd import SystemIdentificationSVD
from tvsclib.expressions.const import Const
from tvsclib.expressions.invert import Invert

def testMixedInversion():
    # Check square
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

    inv_a = Invert(Const(system_a))
    system_inv_a = inv_a.realize()
    rec_inv_a = system_inv_a.to_matrix()
    ref_inv_a = np.linalg.pinv(system_a.to_matrix())

    y_rec = inv_a.compute(u)
    y_ref = ref_inv_a @ u

    assert np.allclose(y_ref, y_rec), "Inversion computation is wrong"
    assert np.allclose(ref_inv_a, rec_inv_a), "Inversion matrix reconstruction is wrong"

    # Check tall
    matrix = np.random.rand(8,5)
    dims_in =  [1, 1, 2, 1]
    dims_out = [2, 3, 1, 2]
    T = ToeplitzOperator(matrix, dims_in, dims_out)
    S = SystemIdentificationSVD(T)

    u = np.random.rand(8,1)

    system_a = MixedSystem(S)

    inv_a = Invert(Const(system_a))
    system_inv_a = inv_a.realize()
    rec_inv_a = system_inv_a.to_matrix()
    ref_inv_a = np.linalg.pinv(system_a.to_matrix())

    y_rec = inv_a.compute(u)
    y_ref = ref_inv_a @ u

    assert np.allclose(y_ref, y_rec), "Inversion computation is wrong"
    assert np.allclose(ref_inv_a, rec_inv_a), "Inversion matrix reconstruction is wrong"

    # Check wide
    matrix = np.random.rand(5,8)
    dims_in =  [2, 3, 1, 2]
    dims_out = [1, 1, 2, 1]
    T = ToeplitzOperator(matrix, dims_in, dims_out)
    S = SystemIdentificationSVD(T)

    u = np.random.rand(5,1)

    system_a = MixedSystem(S)

    inv_a = Invert(Const(system_a))
    system_inv_a = inv_a.realize()
    rec_inv_a = system_inv_a.to_matrix()
    ref_inv_a = np.linalg.pinv(system_a.to_matrix())

    y_rec = inv_a.compute(u)
    y_ref = ref_inv_a @ u

    assert np.allclose(y_ref, y_rec), "Inversion computation is wrong"
    assert np.allclose(ref_inv_a, rec_inv_a), "Inversion matrix reconstruction is wrong"


