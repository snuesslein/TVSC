""" Definition of the transform factory class. """

class TransformFactory(object):
    """ Provides functionality to build a state space transformation operation. """

    def __init__(self):
        """ Constructor. """
        pass

    def get_transform(self,value,transformation,**kwargs):
        """ Builds a state space transformation operation. 
        Note that state space transformations leave the input/output
        behaviour invariant

        Args:
            value (StateSpaceInterface): Entity which shall be transformed in state space
            transformation (TransformationInterface): Transformation that shall be applied
            **kwargs: Additional arguments for the specific transformation

        Returns:
            StateSpaceInterface: Entity that represents a state space transformation
        """
        raise NotImplementedError("Not implemented yet")
