""" Definition of the transpose factory class. """
from tvsclib.causality import Causality
from tvsclib.operations.transpose_strict import TransposeStrict
from tvsclib.operations.transpose_mixed import TransposeMixed

class TransposeFactory(object):
    """ Provides functionality to build a transposition operation. """

    def __init__(self):
        """ Constructor. """
        pass

    def get_transpose(self,value):
        """ Builds a transpose operation. 
        
        Args:
            value: Value that shall be transposed.

        Returns:
            Transpose operation.
        """
        if value.causality is not Causality.MIXED:
            return TransposeStrict(value)
        else:
            return TransposeMixed(value)
