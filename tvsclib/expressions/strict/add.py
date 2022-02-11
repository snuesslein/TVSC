import numpy as np
from scipy.linalg import block_diag
from tvsclib.strict_system import StrictSystem
from tvsclib.stage import Stage

def add(system_lhs:StrictSystem, system_rhs:StrictSystem) -> StrictSystem:
    """add Addition in state space

    Args:
        system_lhs (StrictSystem): left hand side operand
        system_rhs (StrictSystem): right hand side operand

    Returns:
        StrictSystem: Addition result
    """
    k = len(system_rhs.stages)
    stages = []
    for i in range(k):
        A = block_diag(
            system_lhs.stages[i].A_matrix,
            system_rhs.stages[i].A_matrix)
        B = np.vstack([
            system_lhs.stages[i].B_matrix,
            system_rhs.stages[i].B_matrix])
        C = np.hstack([
            system_lhs.stages[i].C_matrix,
            system_rhs.stages[i].C_matrix])
        D = system_lhs.stages[i].D_matrix + system_rhs.stages[i].D_matrix
        stages.append(Stage(A,B,C,D))
    return StrictSystem(causal=system_lhs.causal,stages=stages)