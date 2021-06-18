import numpy as np
from tvsclib.canonical_form import CanonicalForm
from tvsclib.mixed_system import MixedSystem
from tvsclib.toeplitz_operator import ToeplitzOperator
from tvsclib.system_identification_svd import SystemIdentificationSVD
from tvsclib.transformations.reduction import Reduction
from tvsclib.expressions.multiply import Multiply
from tvsclib.expressions.invert import Invert
from tvsclib.expressions.const import Const

def testReductionTransformation():
    matrix = np.random.rand(10,10)
    dims_in =  [2, 4, 1, 3]
    dims_out = [2, 1, 4, 3]
    T = ToeplitzOperator(matrix, dims_in, dims_out)
    S = SystemIdentificationSVD(T,CanonicalForm.BALANCED)

    system_balanced = MixedSystem(S)

    # Invert system twice, expect original system
    c_1 = Const(system_balanced)
    i_1 = Invert(c_1)
    i_2 = Invert(i_1)

    not_reduced = i_2.realize()
    transformation = Reduction()
    reduced = transformation.apply(not_reduced)

    assert ~not_reduced.is_minimal(), "Not reduced system should not be minimal"
    assert reduced.is_minimal(), "Reduced system should be minimal"

    u = np.random.rand(10,1)
    _,y_not_reduced = not_reduced.compute(u)
    _,y_reduced = reduced.compute(u)
    _,y_ref = system_balanced.compute(u)

    assert np.allclose(y_ref, y_not_reduced), "Not reduced system computation wrong"
    assert np.allclose(y_ref, y_reduced), "Reduced system computation wrong"

    matrix_rec = reduced.to_matrix()
    assert np.allclose(matrix, matrix_rec), "Reduced system reconstruction wrong"

    # Cascade system with it's own inverse, expect identity
    c_1 = Const(system_balanced)
    i_1 = Invert(c_1)
    m_1 = Multiply(i_1,c_1)

    not_reduced = m_1.realize()
    transformation = Reduction()
    reduced = transformation.apply(not_reduced)

    assert ~not_reduced.is_minimal(), "Not reduced system should not be minimal"
    assert reduced.is_minimal(), "Reduced system should be minimal"

    u = np.random.rand(10,1)
    _,y_not_reduced = not_reduced.compute(u)
    _,y_reduced = reduced.compute(u)
    y_ref = u

    assert np.allclose(y_ref, y_not_reduced), "Not reduced system computation wrong"
    assert np.allclose(y_ref, y_reduced), "Reduced system computation wrong"

    matrix_rec = reduced.to_matrix()
    matrix_ref = np.identity(matrix.shape[0])
    assert np.allclose(matrix_ref, matrix_rec), "Reduced system reconstruction wrong"