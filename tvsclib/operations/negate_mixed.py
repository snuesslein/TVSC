""" Definition of the negate mixed class. """
from tvsclib.causality import Causality
from tvsclib.realization_mixed import RealizationMixed
from tvsclib.interfaces.statespace_interface import StateSpaceInterface

class NegateMixed(StateSpaceInterface):
    """ Represents a negation operation in state space. """
    
    @property
    def causality(self):
        return self.operand.causality

    def __init__(self,operand:StateSpaceInterface):
        """ Constructor.
        
        Args:
            operand: Operand to negate.
        """
        super().__init__(self._compute_function)
        if operand.causality is not Causality.MIXED:
            raise AttributeError("NegateMixed can not handle strict systems")
        self.operand = operand
    
    def _compute_function(self,u):
        """ Applies a vector to negation result in state space.

        Args:
            u: Vector to be applied.

        Returns: 
            Resulting state vector x and result vector y.
        """
        x,y = self.operand.compute(u)
        return (x,-y)

    def realize(self):
        """ Generates a state space realization of the negation operation. 
        
        Returns:
            Realization of negation operation.
        """
        realization = self.operand.realize()
        return RealizationMixed(
            causal_system = realization.causal_system.neg().realize(),
            anticausal_system = realization.anticausal_system.neg().realize()
        )
