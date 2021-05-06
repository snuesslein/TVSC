""" Definition of the convert strict class. """
from tvsclib.causality import Causality
from tvsclib.realization_mixed import RealizationMixed
from tvsclib.realization_strict import RealizationStrict
from tvsclib.interfaces.statespace_interface import StateSpaceInterface

class ConvertStrict(StateSpaceInterface):
    """ Represents a conversion operation for strict systems.
    A strict system can be converted into a mixed system where
    either the causal or anticausal part is just zero.
    """
    
    @property
    def causality(self):
        return self.into

    def __init__(self,operand:StateSpaceInterface,into:Causality):
        """ Constructor.
        
        Args:
            operand: Operand to convert.
            into: Causality that shall be obtained.
        """
        if operand.causality == Causality.MIXED:
            raise AttributeError("ConvertStrict can not handle mixed systems")
        self.operand = operand
        self.into = into
    
    def compute(self,u):
        """ Applies a vector to conversion result in state space.

        Args:
            u: Vector to be applied.

        Returns: 
            Resulting state vector x and result vector y.
        """
        return self.operand.compute(u)

    def compile(self):
        """ Returns a state space operation that can be directly computed.
        For conversion trivial since it can already be computed.

        Returns:
            Convert operation in state space.
        """
        return self

    def realize(self):
        """ Generates a state space realization of the convert operation. 
        
        Returns:
            Realization of convert operation.
        """
        realization = self.operand.realize()
        if self.into == Causality.MIXED:
            zer = RealizationStrict.zero(
                causal = not realization.causal,
                dims_in = realization.dims_in,
                dims_out = realization.dims_out
            )
            if realization.causal:
                return RealizationMixed(
                    causal_system = realization,
                    anticausal_system = zer
                )
            else:
                return RealizationMixed(
                    causal_system = zer,
                    anticausal_system = realization
                )
        elif self.into == realization.causality:
            return realization
        else:
            raise AttributeError(f"Can not convert from {realization.causality} into {self.into}")