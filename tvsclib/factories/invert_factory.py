""" Definition of the invert factory class. """
import warnings
from tvsclib.causality import Causality
from tvsclib.operations.invert_mixed import InvertMixed
from tvsclib.operations.invert_strict import InvertStrict

class InvertFactory(object):
    """ Provides functionality to build an inversion operation. """

    def __init__(self):
        """ Constructor. """
        pass

    def get_invert(self,operand):
        """ Builds an invert operation. 
        
        Args:
            operand: Operand that shall be inverted.

        Returns:
            Inverted operation.
        """
        if operand.causality is not Causality.MIXED:
            return InvertStrict(operand)
        else:
            return InvertMixed(operand)

