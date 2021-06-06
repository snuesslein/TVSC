import numpy as np
from scipy.linalg import block_diag
from tvsclib.strict_system import StrictSystem
from tvsclib.stage import Stage

def multiply(system_lhs:StrictSystem, system_rhs:StrictSystem) -> StrictSystem:
    """multiply Multiplication in state space

    Args:
        system_lhs (StrictSystem): left hand side operand
        system_rhs (StrictSystem): right hand side operand

    Returns:
        StrictSystem: Multiplication result
    """
    k = len(system_lhs.stages)
    stages = []
    for i in range(k):
        stage_lhs = system_lhs.stages[i]
        stage_rhs = system_rhs.stages[i]
        A = block_diag(stage_rhs.A_matrix, stage_lhs.A_matrix)
        A[
            A.shape[0] - stage_lhs.B_matrix.shape[0]:,
            0:stage_rhs.C_matrix.shape[1]
        ] = stage_lhs.B_matrix @ stage_rhs.C_matrix
        B = np.vstack([
            stage_rhs.B_matrix,
            stage_lhs.B_matrix @ stage_rhs.D_matrix
        ])
        C = np.hstack([
            stage_lhs.D_matrix @ stage_rhs.C_matrix,
            stage_lhs.C_matrix
        ])
        D = stage_lhs.D_matrix @ stage_rhs.D_matrix
        stages.append(Stage(A,B,C,D))
    return StrictSystem(causal=system_lhs.causal,stages=stages)

        