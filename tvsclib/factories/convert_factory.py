""" Definition of the convert factory class. """

class ConvertFactory(object):
    """ Provides functionality to build an conversion operation. """

    def __init__(self):
        """ Constructor. """
        pass

    def get_convert(self,value,into):
        """ Builds an convert operation. 
        
        Args:
            value (StateSpaceInterface): Entity that shall be converted
            into (Enum): Causality type into which entity shall be converted

        Returns:
            StateSpaceInterface: Converted entity
        """
        raise NotImplementedError("Not implemented yet")
