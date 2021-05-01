""" Definition of the negate factory class. """

class NegateFactory(object):
    """ Provides functionality to build a negation operation. """

    def __init__(self):
        """ Constructor. """
        pass

    def get_negate(self,value):
        """ Builds a negate operation. 
        
        Args:
            value (StateSpaceInterface): Entity that shall be negated

        Returns:
            StateSpaceInterface: Negated entity
        """
        raise NotImplementedError("Not implemented yet")
