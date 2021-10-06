import numpy as np
from tvsclib.mixed_system import MixedSystem
from tvsclib.toeplitz_operator import ToeplitzOperator
from tvsclib.system_identification_svd import SystemIdentificationSVD
from tvsclib.transformations.reduction import Reduction

dims_in =  [2, 1, 2, 1]
dims_out = [1, 2, 3, 2]
matrix = np.random.rand(sum(dims_out), sum(dims_in))
T = ToeplitzOperator(matrix, dims_in, dims_out)
S = SystemIdentificationSVD(T)

u = np.random.rand(sum(dims_in),1)

system = MixedSystem(S)
system_causal = system.causal_system
system_anticausal = system.anticausal_system


V,To = system_causal.inner_outer_factorization()
U,R,v,V = system.urv_decomposition()

R_red = Reduction().apply(R)

pass