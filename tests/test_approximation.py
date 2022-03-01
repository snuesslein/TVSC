import numpy as np
from tvsclib.mixed_system import MixedSystem
from tvsclib.toeplitz_operator import ToeplitzOperator
from tvsclib.system_identification_svd import SystemIdentificationSVD
from tvsclib.approximation import Approximation


def testApproxiamtion():
    #simple examle:
    #create a special matrix
    n = 10
    eps = 3
    mat = np.random.rand(2*n,2*n)
    Uc, s, Vch = np.linalg.svd(mat[n:,:n], full_matrices=True)
    Ua, s, Vah = np.linalg.svd(mat[:n,n:], full_matrices=True)
    sc = np.linspace(5,0,n)
    sa = np.linspace(5,0,n)
    mat[n:,:n] = Uc*sc@Vch
    mat[:n,n:] = Ua*sa@Vah

    T = ToeplitzOperator(mat, [n,n], [n,n])
    S = SystemIdentificationSVD(T,epsilon=1e-10)
    system = MixedSystem(S)
    approx =Approximation(system)
    approx_system=approx.get_approxiamtion(eps)

    #check the system by creating a reference matrix
    mat_ref = mat.copy()
    nc = np.count_nonzero(sc>eps)
    na = np.count_nonzero(sa>eps)
    mat_ref[n:,:n] = Uc[:,:nc]*sc[:nc]@Vch[:nc,:]
    mat_ref[:n,n:] = Ua[:,:na]*sa[:na]@Vah[:na,:]

    assert np.allclose(mat_ref,approx_system.to_matrix()), "Approxiamtion does not match reference"

    #more complex system
    eps = 2
    dims_in =  [10]*8
    dims_out = [10]*8
    matrix = np.random.rand(sum(dims_out), sum(dims_in))
    T = ToeplitzOperator(matrix, dims_in, dims_out)
    S = SystemIdentificationSVD(T,epsilon=1e-10)
    system = MixedSystem(S)

    approx =Approximation(system)
    system_approx = approx.get_approxiamtion(eps)


    S_appr = SystemIdentificationSVD(T,epsilon=eps,relative=False)
    system_approx_ident = MixedSystem(S_appr)

    assert np.allclose(system_approx_ident.to_matrix(),system_approx.to_matrix()), \
    "Approxiamtions do not match"
