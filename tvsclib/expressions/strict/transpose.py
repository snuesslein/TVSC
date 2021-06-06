from tvsclib.strict_system import StrictSystem
from tvsclib.stage import Stage

def transpose(system:StrictSystem) -> StrictSystem:
    """transpose Transposition in state space

    Args:
        system (StrictSystem): System to transpose

    Returns:
        StrictSystem: Transposition result
    """
    k = len(system.stages)
    stages = []
    for i in range(k):
        stages.append(Stage(
            system.stages[i].A_matrix.transpose().copy(),
            system.stages[i].C_matrix.transpose().copy(),
            system.stages[i].B_matrix.transpose().copy(),
            system.stages[i].D_matrix.transpose().copy()
        ))
    return StrictSystem(causal=not system.causal, stages=stages)