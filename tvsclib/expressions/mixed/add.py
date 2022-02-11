from tvsclib.mixed_system import MixedSystem
from tvsclib.expressions.strict.add import add as addStrict

def add(system_lhs:MixedSystem, system_rhs:MixedSystem) -> MixedSystem:
    """add Addition of mixed systems

    Args:
        system_lhs (MixedSystem): left hand side operand
        system_rhs (MixedSystem): right hand side operand

    Returns:
        MixedSystem: Addition result
    """
    causal_system = addStrict(
        system_lhs.causal_system,
        system_rhs.causal_system)
    anticausal_system = addStrict(
        system_lhs.anticausal_system,
        system_rhs.anticausal_system)
        
    return MixedSystem(causal_system=causal_system, anticausal_system=anticausal_system)