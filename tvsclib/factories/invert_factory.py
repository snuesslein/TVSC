""" Definition of the invert factory class. """

class InvertFactory(object):
    """ Provides functionality to build an inversion operation. """

    def __init__(self):
        """ Constructor. """
        pass

    def get_invert(self,value):
        """ Builds an invert operation. 
        
        Args:
            value (StateSpaceInterface): Entity that shall be inverted

        Returns:
            StateSpaceInterface: Inverted entity
        """
        raise NotImplementedError("Not implemented yet")
