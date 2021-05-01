""" Definition of the add factory class. """
from tvsclib.causality import Causality
from tvsclib.operations.add_strict import AddStrict

class AddFactory(object):
    """ Provides functionality to build an addition operation. """

    def __init__(self):
        """ Constructor. """
        pass

    def get_add(self,lhs,rhs):
        """ Builds an add operation. 
        
        Args:
            lhs (StateSpaceInterface): Left hand side operator
            rhs (StateSpaceInterface): Right hand side operator

        Returns:
            StateSpaceInterface: Entity that represents an addition in state space
        """
        if lhs.causality == rhs.causality and rhs.causality is not Causality.MIXED:
            return AddStrict(lhs,rhs)
        else:
            raise NotImplementedError("Not implemented yet")
