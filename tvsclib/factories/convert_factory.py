""" Definition of the convert factory class. """
from tvsclib.causality import Causality
from tvsclib.operations.convert_strict import ConvertStrict
from tvsclib.operations.convert_mixed import ConvertMixed

class ConvertFactory(object):
    """ Provides functionality to build an conversion operation. """

    def __init__(self):
        """ Constructor. """
        pass

    def get_convert(self,value:'StateSpaceInterface',into:Causality):
        """ Builds an convert operation. 
        
        Args:
            value: State space object that shall be converted.
            into: Causality type into which state space object shall be converted.

        Returns:
            Convert operation.
        """
        if value.causality is not Causality.MIXED:
            return ConvertStrict(value,into)
        else:
            return ConvertMixed(value,into)
