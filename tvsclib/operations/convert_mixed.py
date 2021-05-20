""" Definition of the convert mixed class. """
from tvsclib.causality import Causality
from tvsclib.interfaces.statespace_interface import StateSpaceInterface

class ConvertMixed(StateSpaceInterface):
    """ Represents a conversion operation for mixed systems.
    A mixed system can only be converted into a mixed system.
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
        super().__init__(self._compute_function)
        if operand.causality is not Causality.MIXED:
            raise AttributeError("ConvertMixed can not handle strict systems")
        self.operand = operand
        self.into = into
    
    def _compute_function(self,u):
        """ Applies a vector to conversion result in state space.

        Args:
            u: Vector to be applied.

        Returns: 
            Resulting state vector x and result vector y.
        """
        return self.operand.compute(u)

    def realize(self):
        """ Generates a state space realization of the convert operation. 
        
        Returns:
            Realization of convert operation.
        """
        realization = self.operand.realize()
        if self.into == Causality.MIXED:
            return realization
        else:
            raise AttributeError(f"Can not convert from {realization.causality} into {self.into}")