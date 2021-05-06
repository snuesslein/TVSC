""" Definition of the negate factory class. """
from tvsclib.causality import Causality
from tvsclib.operations.negate_strict import NegateStrict
from tvsclib.operations.negate_mixed import NegateMixed

class NegateFactory(object):
    """ Provides functionality to build a negation operation. """

    def __init__(self):
        """ Constructor. """
        pass

    def get_negate(self,value):
        """ Builds a negate operation. 
        
        Args:
            value: Value that shall be negated.

        Returns:
            Negate operation.
        """
        if value.causality is not Causality.MIXED:
            return NegateStrict(value)
        else:
            return NegateMixed(value)
