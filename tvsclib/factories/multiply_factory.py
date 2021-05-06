""" Definition of the multiply factory class. """
from tvsclib.causality import Causality
from tvsclib.operations.multiply_strict import MultiplyStrict

class MultiplyFactory(object):
    """ Provides functionality to build a multiplication operation. """

    def __init__(self):
        """ Constructor. """
        pass

    def get_multiply(self,lhs,rhs):
        """ Builds an multiplication operation. 
        
        Args:
            lhs: Left hand side operator.
            rhs: Right hand side operator.

        Returns:
            Multiply operation.
        """
        if lhs.causality == rhs.causality and rhs.causality is not Causality.MIXED:
            return MultiplyStrict(lhs,rhs)
        else:
            raise NotImplementedError("Not implemented yet")
