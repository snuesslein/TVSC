import numpy as np
from tvsclib.mixed_system import MixedSystem
from tvsclib.toeplitz_operator import ToeplitzOperator
from tvsclib.system_identification_svd import SystemIdentificationSVD
from tvsclib.transformations.reduction import Reduction

dims_in =  [1]*8
dims_out = [1]*8
matrix = np.random.rand(sum(dims_out), sum(dims_in))
matrix = matrix - np.tril(matrix,-2) - np.triu(matrix,2) # Banded shape
matrix = np.linalg.inv(matrix)                           # Obscure structure
T = ToeplitzOperator(matrix, dims_in, dims_out)
S = SystemIdentificationSVD(T)

system = MixedSystem(S)
system_causal = system.causal_system
system_anticausal = system.anticausal_system

print(f"Causal state dimensions: {system_causal.dims_state}")
print(f"Anticausal state dimensions: {system_anticausal.dims_state}")


V,To = system_causal.inner_outer_factorization()
U,R,v,V = system.urv_decomposition()

R_red = Reduction().apply(R)

pass