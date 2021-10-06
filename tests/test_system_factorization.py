import numpy as np
from tvsclib.mixed_system import MixedSystem
from tvsclib.toeplitz_operator import ToeplitzOperator
from tvsclib.system_identification_svd import SystemIdentificationSVD

def testSystemFactorization():
    dims_in =  [2, 1, 2, 1]
    dims_out = [1, 2, 3, 2]
    matrix = np.random.rand(sum(dims_out), sum(dims_in))
    T = ToeplitzOperator(matrix, dims_in, dims_out)
    S = SystemIdentificationSVD(T)

    u = np.random.rand(sum(dims_in),1)

    system = MixedSystem(S)
    system_causal = system.causal_system

    U,R,v,V = system.urv_decomposition()


    assert np.allclose(U.to_matrix() @ R.to_matrix() @ v.to_matrix() @ V.to_matrix(), matrix), "URV decomposition is wrong"
    assert np.allclose(U.transpose().to_matrix() @ U.to_matrix(), np.eye(sum(U.dims_in))), "U is not unitary"
    assert np.allclose(v.transpose().to_matrix() @ v.to_matrix(), np.eye(sum(v.dims_in))), "v is not unitary"
    assert np.allclose(V.transpose().to_matrix() @ V.to_matrix(), np.eye(sum(V.dims_in))), "V is not unitary"
    assert np.allclose(R.arrow_reversal().to_matrix() @ R.to_matrix(), np.eye(sum(R.dims_in))), "R inverse is wrong"

