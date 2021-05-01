from tvsclib.interfaces.statespace_interface import StateSpaceInterface
from tvsclib.interfaces.separation_interface import SeparationInterface
from tvsclib.interfaces.realization_interface import RealizationInterface
from tvsclib.factories import (
    AddFactory as _AddFactory, 
    NegateFactory as _NegateFactory,
    MultiplyFactory as _MultiplyFactory,
    InvertFactory as _InvertFactory,
    TransposeFactory as _TransposeFactory,
    ConvertFactory as _ConvertFactory)

# Initialize StateSpaceInterface
StateSpaceInterface.add_factory = _AddFactory()
StateSpaceInterface.negate_factory = _NegateFactory()
StateSpaceInterface.multiply_factory = _MultiplyFactory()
StateSpaceInterface.invert_factory = _InvertFactory()
StateSpaceInterface.transpose_factory = _TransposeFactory()
StateSpaceInterface.convert_factory = _ConvertFactory()
