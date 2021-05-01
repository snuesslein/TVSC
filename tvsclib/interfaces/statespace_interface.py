""" Definition of the state  space interface. """
import abc
from tvsclib.causality import Causality

class StateSpaceInterface(object):
    __metaclass__ = abc.ABCMeta
    """ Classes which inherit this interface can represent
    operations and entities in state space.

    Attributes:
        add_factory (AddFactory): Factory used to build add operations
        negate_factory (NegateFactory): Factory used to build negate operations
        multiply_factory (MultiplyFactory): Factory used to build multiply operations
        invert_factory (InvertFactory): Factory used to build invert operations
        transpose_factory (TransposeFactory): Factory used to build transpose operations
        convert_factory (ConvertFactory): Factory used to build convert operations
    """

    add_factory = None
    negate_factory = None
    multiply_factory = None
    invert_factory = None
    transpose_factory = None

    def __init__(self):
        """ Constructor. """  
        pass

    @abc.abstractmethod
    def compute(self,u):
        """ Computes the result of a vector applied to this state space entity.

        Args:
            u (float[]): Vector which is applied

        Returns:
            x,y: Resulting state vector x and result vector y
        """
        pass

    @abc.abstractmethod
    def compile(self):
        """ Compiles state space entity into more basic, faster computable operations.

        Returns:
            StateSpaceInterface: State space entity in the form of fast computable operations
        """
        pass 

    @abc.abstractmethod
    def realize(self):
        """ Generates a concrete realization of the state space entity.

        Returns:
            RealizationInterface: Realization of the state space entity
        """

    @abc.abstractproperty
    def causality(self):
        """ Either causal, anticausal or mixed. """
        pass

    def add(self,rhs):
        """ Addition in state space.

        Args:
            rhs (StateSpaceInterface): Right hand side of the addition operation
        
        Returns:
            StateSpaceInterface: Addition result in state space
        """      
        return StateSpaceInterface.add_factory.get_add(self,rhs)
    
    def mul(self,rhs):
        """ Multiplication in state space.

        Args:
            rhs (StateSpaceInterface): Right hand side of the multiplication operation
        
        Returns:
            StateSpaceInterface: Multiplication result in state space
        """
        raise NotImplementedError("Not implemented yet")
    
    def neg(self):
        """ Negation in state space.
        
        Returns:
            StateSpaceInterface: Negation result in state space
        """
        raise NotImplementedError("Not implemented yet")
    
    def inv(self):
        """ Inversion in state space.
        
        Returns:
            StateSpaceInterface: Inversion result in state space
        """
        raise NotImplementedError("Not implemented yet")
    
    def transpose(self):
        """ Transposition in state space.
        
        Returns:
            StateSpaceInterface: Transposition result in state space
        """
        raise NotImplementedError("Not implemented yet")

    def convert(self,into):
        """ Conversion of a state space entity to a differnt causality type.

        Args:
            into (Enum): The causality into which the state space entity shall be converted
        
        Returns:
            StateSpaceInterface: Conversion result in state space
        """
        raise NotImplementedError("Not implemented yet")
    
    def transform(self,transformation,**kwargs):
        """ State transformation in state space.
        
        Args:
            transformation (string): Name of the transformation
            **kwargs: Arguments for the specific transformation

        Returns:
            StateSpaceInterface: A entity with the same input output behaviour but different state space
        """
        raise NotImplementedError("Not implemented yet")