from tvsclib.strict_system import StrictSystem

def negate(system:StrictSystem) -> StrictSystem:
    """negate Negation in state space

    Args:
        system (StrictSystem): System to negate

    Returns:
        StrictSystem: Negation result
    """
    negated_system = system.copy()
    k = len(negated_system.stages)
    for i in range(k):
        negated_system.stages[i].C_matrix = -negated_system.stages[i].C_matrix
        negated_system.stages[i].D_matrix = -negated_system.stages[i].D_matrix
    return negated_system