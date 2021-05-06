""" Definition of the convert factory class. """

class ConvertFactory(object):
    """ Provides functionality to build an conversion operation. """

    def __init__(self):
        """ Constructor. """
        pass

    def get_convert(self,value,into):
        """ Builds an convert operation. 
        
        Args:
            value: Entity that shall be converted.
            into: Causality type into which entity shall be converted.

        Returns:
            Convert operation.
        """
        raise NotImplementedError("Not implemented yet")
