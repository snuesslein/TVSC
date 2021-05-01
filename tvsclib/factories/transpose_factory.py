""" Definition of the transpose factory class. """

class TransposeFactory(object):
    """ Provides functionality to build a transposition operation. """

    def __init__(self):
        """ Constructor. """
        pass

    def get_transpose(self,value):
        """ Builds a transpose operation. 
        
        Args:
            value (StateSpaceInterface): Entity that shall be transposed

        Returns:
            StateSpaceInterface: Transposed entity
        """
        raise NotImplementedError("Not implemented yet")
