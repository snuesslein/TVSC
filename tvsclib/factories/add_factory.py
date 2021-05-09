""" Definition of the add factory class. """
from tvsclib.causality import Causality
from tvsclib.operations.add_strict import AddStrict
from tvsclib.operations.add_mixed import AddMixed

class AddFactory(object):
    """ Provides functionality to build an addition operation. """

    def __init__(self):
        """ Constructor. """
        pass

    def get_add(self,lhs:'StateSpaceInterface',rhs:'StateSpaceInterface'):
        """ Builds an add operation. 
        
        Args:
            lhs: Left hand side operator.
            rhs: Right hand side operator.

        Returns:
            Addition operation.
        """
        if lhs.causality == rhs.causality and rhs.causality is not Causality.MIXED:
            return AddStrict(lhs,rhs)
        else:
            return AddMixed(
                lhs.convert(Causality.MIXED),
                rhs.convert(Causality.MIXED)
            )
