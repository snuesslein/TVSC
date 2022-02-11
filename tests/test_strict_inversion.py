import numpy as np
from tvsclib.mixed_system import MixedSystem
from tvsclib.toeplitz_operator import ToeplitzOperator
from tvsclib.system_identification_svd import SystemIdentificationSVD
from tvsclib.expressions.const import Const
from tvsclib.expressions.invert import Invert
from tvsclib.expressions.transpose import Transpose

def testStrictInversion():
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

    system_a = MixedSystem(S).causal_system

    # Check causal
    inv_a = Invert(Const(system_a))
    system_inv_a = inv_a.realize()
    rec_inv_a = system_inv_a.to_matrix()
    ref_inv_a = np.linalg.pinv(system_a.to_matrix())

    y_rec = inv_a.compute(u)
    y_ref = ref_inv_a @ u

    assert np.allclose(y_ref, y_rec), "Inversion computation is wrong"
    assert np.allclose(ref_inv_a, rec_inv_a), "Inversion matrix reconstruction is wrong"

    # Check anticausal
    inv_b = Invert(Transpose(Const(system_a)))
    system_inv_b = inv_b.realize()
    rec_inv_b = system_inv_b.to_matrix()
    ref_inv_b = np.linalg.pinv(system_a.to_matrix().transpose())

    y_rec = inv_b.compute(u)
    y_ref = rec_inv_b @ u

    assert np.allclose(y_ref, y_rec), "Inversion computation is wrong"
    assert np.allclose(ref_inv_b, rec_inv_b), "Inversion matrix reconstruction is wrong"

    # Tall matrix
    matrix = np.random.rand(8,6)
    dims_in =  [2, 1, 2, 1]
    dims_out = [1, 2, 3, 2]
    T = ToeplitzOperator(matrix, dims_in, dims_out)
    S = SystemIdentificationSVD(T)

    u = np.random.rand(8,1)

    system_a = MixedSystem(S).causal_system
    inv_a = Invert(Const(system_a))
    system_inv_a = inv_a.realize()
    rec_inv_a = system_inv_a.to_matrix()
    ref_inv_a = np.linalg.pinv(system_a.to_matrix())

    y_rec = inv_a.compute(u)
    y_ref = ref_inv_a @ u

    assert np.allclose(y_ref, y_rec), "Inversion computation is wrong"
    assert np.allclose(ref_inv_a, rec_inv_a), "Inversion matrix reconstruction is wrong"

    inv_b = Invert(Transpose(Const(system_a)))
    system_inv_b = inv_b.realize()
    rec_inv_b = system_inv_b.to_matrix()
    ref_inv_b = np.linalg.pinv(system_a.to_matrix().transpose())
    
    u = np.random.rand(6,1)
    
    y_rec = inv_b.compute(u)
    y_ref = ref_inv_b @ u

    assert np.allclose(y_ref, y_rec), "Inversion computation is wrong"
    assert np.allclose(ref_inv_b, rec_inv_b), "Inversion matrix reconstruction is wrong"

    # Wide matrix
    matrix = np.random.rand(6,8)
    dims_in =  [1, 2, 3, 2]
    dims_out = [2, 1, 2, 1]
    T = ToeplitzOperator(matrix, dims_in, dims_out)
    S = SystemIdentificationSVD(T)

    u = np.random.rand(6,1)

    system_a = MixedSystem(S).causal_system
    inv_a = Invert(Const(system_a))
    system_inv_a = inv_a.realize()
    rec_inv_a = system_inv_a.to_matrix()
    ref_inv_a = np.linalg.pinv(system_a.to_matrix())

    y_rec = inv_a.compute(u)
    y_ref = ref_inv_a @ u

    assert np.allclose(y_ref, y_rec), "Inversion computation is wrong"
    assert np.allclose(ref_inv_a, rec_inv_a), "Inversion matrix reconstruction is wrong"

    inv_b = Invert(Transpose(Const(system_a)))
    system_inv_b = inv_b.realize()
    rec_inv_b = system_inv_b.to_matrix()
    ref_inv_b = np.linalg.pinv(system_a.to_matrix().transpose())
    
    u = np.random.rand(8,1)
    
    y_rec = inv_b.compute(u)
    y_ref = ref_inv_b @ u

    assert np.allclose(y_ref, y_rec), "Inversion computation is wrong"
    assert np.allclose(ref_inv_b, rec_inv_b), "Inversion matrix reconstruction is wrong"


