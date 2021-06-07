from tvsclib.mixed_system import MixedSystem
from tvsclib.expressions.multiply import Multiply
from tvsclib.expressions.transpose import Transpose
from tvsclib.expressions.const import Const

def invert(system:MixedSystem) -> Multiply:
    return Transpose(Const(system)) # Just for debugging
    #raise NotImplementedError()