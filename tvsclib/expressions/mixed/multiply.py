import numpy as np
from tvsclib.strict_system import StrictSystem
from tvsclib.mixed_system import MixedSystem
from tvsclib.stage import Stage
from tvsclib.expressions.strict.add import add as addStrict
from tvsclib.expressions.mixed.add import add as addMixed
from tvsclib.expressions.strict.multiply import multiply as multiplyStrict

def multiply(system_lhs:MixedSystem, system_rhs:MixedSystem) -> MixedSystem:
    """multiply Multiplication of mixed systems

    Args:
        system_lhs (MixedSystem): left hand side operand
        system_rhs (MixedSystem): right hand side operand

    Returns:
        MixedSystem: Multiplication result
    """
    result_causal = multiplyStrict(system_lhs.causal_system, system_rhs.causal_system)
    result_anticausal = multiplyStrict(system_lhs.anticausal_system, system_rhs.anticausal_system)
    result_mixed = _causal_times_anticausal(system_lhs.causal_system, system_rhs.anticausal_system)
    result_mixed = addMixed(result_mixed, _anticausal_times_causal(
        system_lhs.anticausal_system, system_rhs.causal_system))
    return MixedSystem(
        causal_system=addStrict(result_causal, result_mixed.causal_system),
        anticausal_system=addStrict(result_anticausal, result_mixed.anticausal_system))

def _anticausal_times_causal(anticausal_system:StrictSystem, causal_system:StrictSystem) -> MixedSystem:
    k = len(causal_system.stages)
    M = [np.zeros((0,0))]*(k+1)
    for i in range(k-1,-1,-1):
        M[i] = anticausal_system.stages[i].B_matrix @ causal_system.stages[i].C_matrix \
            + anticausal_system.stages[i].A_matrix @ M[i+1] @ causal_system.stages[i].A_matrix
    # Compute strictly anticasual part of multiplication result
    stages_anticausal = []
    for i in range(k):
        stages_anticausal.append(Stage(
            anticausal_system.stages[i].A_matrix,
            anticausal_system.stages[i].A_matrix @ M[i+1] @ causal_system.stages[i].B_matrix,
            anticausal_system.stages[i].C_matrix,
            anticausal_system.stages[i].C_matrix @ M[i+1] @ causal_system.stages[i].B_matrix))
    result_anticausal = StrictSystem(
        causal=False,
        stages=stages_anticausal)
    # Compute strictly causal part of multiplication result
    stages_causal = []
    for i in range(k):
        stages_causal.append(Stage(
            causal_system.stages[i].A_matrix,
            causal_system.stages[i].B_matrix,
            anticausal_system.stages[i].C_matrix @ M[i+1] @ causal_system.stages[i].A_matrix,
            np.zeros((anticausal_system.stages[i].C_matrix.shape[0], causal_system.stages[i].B_matrix.shape[1]))))
    result_causal = StrictSystem(
        causal=True,
        stages=stages_causal)
    # Causal pass-through part as anticausal system
    stages_pass_causal = []
    for i in range(k):
        stages_pass_causal.append(Stage(
            np.zeros((0,0)),
            np.zeros((0,causal_system.stages[i].D_matrix.shape[1])),
            np.zeros((causal_system.stages[i].D_matrix.shape[0],0)),
            causal_system.stages[i].D_matrix))
    system_causal_pass_as_anticausal = StrictSystem(
        causal=False,
        stages=stages_pass_causal)
    # Anticausal pass-through part as causal and anticausal system
    stages_pass_anticausal = []
    for i in range(k):
        stages_pass_anticausal.append(Stage(
            np.zeros((0,0)),
            np.zeros((0,anticausal_system.stages[i].D_matrix.shape[1])),
            np.zeros((anticausal_system.stages[i].D_matrix.shape[0],0)),
            anticausal_system.stages[i].D_matrix))
    system_anticausal_pass_as_causal = StrictSystem(
        causal=True,
        stages=stages_pass_anticausal)
    system_anticausal_pass_as_anticausal = StrictSystem(
        causal=False,
        stages=stages_pass_anticausal)
    # Adding product of pass-through parts
    result_anticausal = addStrict(
        result_anticausal,
        multiplyStrict(system_anticausal_pass_as_anticausal, system_causal_pass_as_anticausal))
    # Anticausal system without pass-through part
    stages_pure_anticausal = []
    for i in range(k):
        stages_pure_anticausal.append(Stage(
            anticausal_system.stages[i].A_matrix,
            anticausal_system.stages[i].B_matrix,
            anticausal_system.stages[i].C_matrix,
            np.zeros(anticausal_system.stages[i].D_matrix.shape)))
    system_pure_anticausal = StrictSystem(
        causal=False,
        stages=stages_pure_anticausal)
    # Adding product of pure anticausal and causal pass-through system
    result_anticausal = addStrict(
        result_anticausal,
        multiplyStrict(system_pure_anticausal, system_causal_pass_as_anticausal))
    # Causal system without pass-through part
    stages_pure_causal = []
    for i in range(k):
        stages_pure_causal.append(Stage(
            causal_system.stages[i].A_matrix,
            causal_system.stages[i].B_matrix,
            causal_system.stages[i].C_matrix,
            np.zeros(causal_system.stages[i].D_matrix.shape)))
    system_pure_causal = StrictSystem(
        causal=True,
        stages=stages_pure_causal)
    # Adding product of anticausal pass-through and pure causal system
    result_causal = addStrict(
        result_causal,
        multiplyStrict(system_anticausal_pass_as_causal, system_pure_causal))
    return MixedSystem(causal_system=result_causal, anticausal_system=result_anticausal)

def _causal_times_anticausal(causal_system:StrictSystem, anticausal_system:StrictSystem) -> MixedSystem:
    k = len(causal_system.stages)
    M = [np.zeros((0,0))]
    for i in range(k):
        M.append(
            causal_system.stages[i].B_matrix @ anticausal_system.stages[i].C_matrix
            + causal_system.stages[i].A_matrix @ M[i] @ anticausal_system.stages[i].A_matrix
        )
    # Compute strictly causal part of multiplication result
    causal_stages = []
    for i in range(k):
        causal_stages.append(Stage(
            causal_system.stages[i].A_matrix,
            causal_system.stages[i].A_matrix @ M[i] @ anticausal_system.stages[i].B_matrix,
            causal_system.stages[i].C_matrix,
            causal_system.stages[i].C_matrix @ M[i] @ anticausal_system.stages[i].B_matrix))
    result_causal = StrictSystem(
        causal=True,
        stages=causal_stages)
    # Compute strictly anticausal part of multiplication result
    anticausal_stages = []
    for i in range(k):
        anticausal_stages.append(Stage(
            anticausal_system.stages[i].A_matrix,
            anticausal_system.stages[i].B_matrix,
            causal_system.stages[i].C_matrix @ M[i] @ anticausal_system.stages[i].A_matrix,
            np.zeros((causal_system.stages[i].C_matrix.shape[0], anticausal_system.stages[i].B_matrix.shape[1]))))
    result_anticausal = StrictSystem(
        causal=False,
        stages=anticausal_stages)
    # Causal pass-through part as a anticausal system
    stages_pass_causal = []
    for i in range(k):
        stages_pass_causal.append(Stage(
            np.zeros((0,0)),
            np.zeros((0,causal_system.stages[i].D_matrix.shape[1])),
            np.zeros((causal_system.stages[i].D_matrix.shape[0],0)),
            causal_system.stages[i].D_matrix))
    system_causal_pass_as_anticausal = StrictSystem(
        causal=False,
        stages=stages_pass_causal)
    # Anticausal pass-through part as causal and anticausal system
    stages_pass_anticausal = []
    for i in range(k):
        stages_pass_anticausal.append(Stage(
            np.zeros((0,0)),
            np.zeros((0,anticausal_system.stages[i].D_matrix.shape[1])),
            np.zeros((anticausal_system.stages[i].D_matrix.shape[0],0)),
            anticausal_system.stages[i].D_matrix
        ))
    system_anticausal_pass_as_causal = StrictSystem(
        causal=True,
        stages=stages_pass_anticausal)
    system_anticausal_pass_as_anticausal = StrictSystem(
        causal=False,
        stages=stages_pass_anticausal)
    # Adding product of pass-through parts
    result_anticausal = addStrict(
        result_anticausal,
        multiplyStrict(system_causal_pass_as_anticausal, system_anticausal_pass_as_anticausal))
    # Anticausal system without pass-through part
    stages_pure_anticausal = []
    for i in range(k):
        stages_pure_anticausal.append(Stage(
            anticausal_system.stages[i].A_matrix,
            anticausal_system.stages[i].B_matrix,
            anticausal_system.stages[i].C_matrix,
            np.zeros(anticausal_system.stages[i].D_matrix.shape)
        ))
    system_pure_anticausal = StrictSystem(
        causal=False,
        stages=stages_pure_anticausal)
    # Adding product of causal pass-through system and pure anticausal system
    result_anticausal = addStrict(
        result_anticausal,
        multiplyStrict(system_causal_pass_as_anticausal, system_pure_anticausal))
    # Causal system without pass-through part
    stages_pure_causal = []
    for i in range(k):
        stages_pure_causal.append(Stage(
            causal_system.stages[i].A_matrix,
            causal_system.stages[i].B_matrix,
            causal_system.stages[i].C_matrix,
            np.zeros(causal_system.stages[i].D_matrix.shape)
        ))
    system_pure_causal = StrictSystem(
        causal=True,
        stages=stages_pure_causal)
    # Adding product of pure causal system and anticausal pass-through system
    result_causal = addStrict(
        result_causal,
        multiplyStrict(system_pure_causal, system_anticausal_pass_as_causal))
    return MixedSystem(causal_system=result_causal, anticausal_system=result_anticausal)