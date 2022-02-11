from tvsclib.system_interface import SystemInterface
from tvsclib.strict_system import StrictSystem
from tvsclib.mixed_system import MixedSystem

def convert(system:SystemInterface, into:type) -> SystemInterface:
    if into is type(system):
        return system
    elif into is MixedSystem and type(system) is StrictSystem:
        zero_system = StrictSystem.zero(not system.causal, system.dims_in, system.dims_out)
        if system.causal:
            return MixedSystem(causal_system=system, anticausal_system=zero_system)
        return MixedSystem(causal_system=zero_system, anticausal_system=system)
    raise AttributeError(f"Can not convert {type(system)} into {into}")