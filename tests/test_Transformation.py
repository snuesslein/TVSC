import numpy as np
from tvsclib.canonical_form import CanonicalForm
from tvsclib.mixed_system import MixedSystem
from tvsclib.toeplitz_operator import ToeplitzOperator
from tvsclib.system_identification_svd import SystemIdentificationSVD

from tvsclib.transformations.reduction import Reduction
from tvsclib.transformations.input_normal import InputNormal
from tvsclib.transformations.output_normal import OutputNormal

def testSystem():
    dims_in =  [2, 1, 2, 1, 5, 2,10, 3, 2, 1, 3, 2, 4, 2, 5,20,30,10,10,10,15]
    dims_out = [1, 2, 1, 2, 5, 2, 7, 3, 2, 1, 5, 7, 2, 1, 2,20,30,10,10,10,15]
    matrix = np.random.rand(sum(dims_out), sum(dims_in))
    T = ToeplitzOperator(matrix, dims_in, dims_out)
    S = SystemIdentificationSVD(T,epsilon=1e-10)
    system = MixedSystem(S)
    system_causal = system.causal_system
    system_anticausal = system.anticausal_system

    #check make input InputNormal
    sys_causal_inp = InputNormal().apply(system_causal)
    sys_anticausal_inp = InputNormal().apply(system_anticausal)

    assert sys_causal_inp.is_input_normal(), "causal system is not input normal"
    assert sys_anticausal_inp.is_input_normal(), "anticausal system is not input normal"

    assert not sys_causal_inp.stages is system_causal.stages \
        and not sys_causal_inp.stages[1] is system_causal.stages[1] \
        and not sys_causal_inp.stages[1].A_matrix is system_causal.stages[1].A_matrix,\
        "input normal anticausal system is not copied"
    assert not sys_anticausal_inp.stages is system_anticausal.stages \
        and not sys_anticausal_inp.stages[1] is system_anticausal.stages[1] \
        and not sys_anticausal_inp.stages[1].A_matrix is system_anticausal.stages[1].A_matrix,\
        "input normal anticausal system is not copied"


    assert np.allclose(sys_causal_inp.to_matrix(),system_causal.to_matrix()),\
        "input normal causal system is not equivalent"
    assert np.allclose(sys_anticausal_inp.to_matrix(),system_anticausal.to_matrix()),\
        "input normal anticausal system is not equivalent"


    #check make output InputNormal
    sys_causal_out = OutputNormal().apply(system_causal)
    sys_anticausal_out = OutputNormal().apply(system_anticausal)

    assert sys_causal_out.is_output_normal(), "causal system is not output normal"
    assert sys_anticausal_out.is_output_normal(), "anticausal system is not output normal"

    assert not sys_causal_out.stages is system_causal.stages \
        and not sys_causal_out.stages[1] is system_causal.stages[1] \
        and not sys_causal_out.stages[1].A_matrix is system_causal.stages[1].A_matrix,\
        "output normal anticausal system is not copied"
    assert not sys_anticausal_out.stages is system_anticausal.stages \
        and not sys_anticausal_out.stages[1] is system_anticausal.stages[1] \
        and not sys_anticausal_out.stages[1].A_matrix is system_anticausal.stages[1].A_matrix,\
        "output normal anticausal system is not copied"


    assert np.allclose(sys_causal_out.to_matrix(),system_causal.to_matrix()),\
        "output normal causal system is not equivalent"
    assert np.allclose(sys_anticausal_out.to_matrix(),system_anticausal.to_matrix()),\
        "output normal anticausal system is not equivalent"
