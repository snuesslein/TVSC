from tvsclib.mixed_system import MixedSystem
from tvsclib.expressions.strict.transpose import transpose as transposeStrict

def transpose(system:MixedSystem) -> MixedSystem:
    """transpose Transposition in state space

    Args:
        system (MixedSystem): System to transpose

    Returns:
        MixedSystem: Transposition result
    """
    return MixedSystem(
        causal_system=transposeStrict(system.anticausal_system),
        anticausal_system=transposeStrict(system.causal_system))