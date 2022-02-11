from tvsclib.strict_system import StrictSystem
from tvsclib.stage import Stage

def transpose(system:StrictSystem) -> StrictSystem:
    """transpose Transposition in state space

    Args:
        system (StrictSystem): System to transpose

    Returns:
        StrictSystem: Transposition result 
    """
    return system.transpose()