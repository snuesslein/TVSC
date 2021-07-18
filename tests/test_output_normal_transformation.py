import numpy as np
import pytest
from tvsclib.canonical_form import CanonicalForm
from tvsclib.mixed_system import MixedSystem
from tvsclib.toeplitz_operator import ToeplitzOperator
from tvsclib.system_identification_svd import SystemIdentificationSVD
from tvsclib.transformations.output_normal import OutputNormal
from tvsclib.expressions.add import Add
from tvsclib.expressions.const import Const 

def testOutputNormalTransformation():
    matrix = np.random.rand(6,6)
    dims_in =  [2, 1, 2, 1]
    dims_out = [1, 2, 1, 2]
    T = ToeplitzOperator(matrix, dims_in, dims_out)
    S = SystemIdentificationSVD(T,CanonicalForm.BALANCED)

    u = np.random.rand(6,1)
    y_ref = matrix@u

    system_balanced = MixedSystem(S)
    transformation = OutputNormal()
    system_onf = transformation.apply(system_balanced)
    
    const_1 = Const(system_balanced)
    const_2 = Const(system_onf)
    add = Add(const_1, const_2)
    add_realization = add.realize()
    with pytest.raises(Exception):
        crash = OutputNormal().apply(add_realization)

    x, y_rec = system_onf.compute(u)
    assert np.allclose(y_ref, y_rec), "Transformed system computation is wrong"

    stages_causal = system_onf.causal_system.stages
    stages_anticausal = system_onf.anticausal_system.stages

    for i in range(len(stages_causal)):
        I = stages_causal[i].A_matrix.transpose() @ stages_causal[i].A_matrix\
            + stages_causal[i].C_matrix.transpose() @ stages_causal[i].C_matrix
        I_ref = np.eye(I.shape[0])
        assert np.allclose(I_ref, I), "Causal system not in ONF"
    
    
    for i in range(len(stages_anticausal)):
        I = stages_anticausal[i].A_matrix.transpose() @ stages_anticausal[i].A_matrix\
            + stages_anticausal[i].C_matrix.transpose() @ stages_anticausal[i].C_matrix
        I_ref = np.eye(I.shape[0])
        assert np.allclose(I_ref, I), "Anticausal system not in ONF"

    # Only causal
    system_balanced = MixedSystem(S).causal_system
    transformation = OutputNormal()
    system_onf = transformation.apply(system_balanced)
    y_ref = system_balanced.to_matrix()@u
    x, y_rec = system_onf.compute(u)
    assert np.allclose(y_ref, y_rec), "Transformed system computation is wrong"

    stages_causal = system_onf.stages

    for i in range(len(stages_causal)):
        I = stages_causal[i].A_matrix.transpose() @ stages_causal[i].A_matrix\
            + stages_causal[i].C_matrix.transpose() @ stages_causal[i].C_matrix
        I_ref = np.eye(I.shape[0])
        assert np.allclose(I_ref, I), "Causal system not in ONF"
    
    # Only anticausal
    system_balanced = MixedSystem(S).anticausal_system
    transformation = OutputNormal()
    system_onf = transformation.apply(system_balanced)
    y_ref = system_balanced.to_matrix()@u
    x, y_rec = system_onf.compute(u)
    assert np.allclose(y_ref, y_rec), "Transformed system computation is wrong"

    stages_anticausal = system_onf.stages

    for i in range(len(stages_anticausal)):
        I = stages_anticausal[i].A_matrix.transpose() @ stages_anticausal[i].A_matrix\
            + stages_anticausal[i].C_matrix.transpose() @ stages_anticausal[i].C_matrix
        I_ref = np.eye(I.shape[0])
        assert np.allclose(I_ref, I), "Anticausal system not in ONF"
    