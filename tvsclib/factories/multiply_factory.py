""" Definition of the multiply factory class. """

class MultiplyFactory(object):
    """ Provides functionality to build a multiplication operation. """

    def __init__(self):
        """ Constructor. """
        pass

    def get_multiply(self,lhs,rhs):
        """ Builds an multiplication operation. 
        
        Args:
            lhs (StateSpaceInterface): Left hand side operator
            rhs (StateSpaceInterface): Right hand side operator

        Returns:
            StateSpaceInterface: Entity that represents an multiplication in state space
        """
        raise NotImplementedError("Not implemented yet")
