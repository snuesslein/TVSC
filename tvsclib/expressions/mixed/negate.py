from tvsclib.mixed_system import MixedSystem
from tvsclib.expressions.strict.negate import negate as negateStrict

def negate(system:MixedSystem) -> MixedSystem:
    """negate Negation in state space

    Args:
        system (MixedSystem): System to negate

    Returns:
        MixedSystem: Negation result
    """
    return MixedSystem(
        causal_system=negateStrict(system.causal_system),
        anticausal_system=negateStrict(system.anticausal_system))