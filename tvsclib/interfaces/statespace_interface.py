""" Definition of the state  space interface. """
import abc
from tvsclib.causality import Causality

class StateSpaceInterface(object):
    __metaclass__ = abc.ABCMeta
    """ Classes which inherit this interface can represent
    operations and entities in state space.
    """

    _add_factory = None
    _negate_factory = None
    _multiply_factory = None
    _invert_factory = None
    _transpose_factory = None

    @property
    def add_factory():
        """Add Factory object."""
        return StateSpaceInterface._add_factory

    @add_factory.setter
    def add_factory(value:'AddFactory'):
        StateSpaceInterface._add_factory = value

    @property
    def negate_factory():
        """Negate Factory object."""
        return StateSpaceInterface._negate_factory

    @negate_factory.setter
    def negate_factory(value:'NegateFactory'):
        StateSpaceInterface._negate_factory = value

    @property
    def multiply_factory():
        """Multiply Factory object."""
        return StateSpaceInterface._multiply_factory

    @multiply_factory.setter
    def multiply_factory(value:'MultiplyFactory'):
        StateSpaceInterface._multiply_factory = value

    @property
    def invert_factory():
        """Invert Factory object."""
        return StateSpaceInterface._invert_factory

    @invert_factory.setter
    def invert_factory(value:'InvertFactory'):
        StateSpaceInterface._invert_factory = value

    @property
    def transpose_factory():
        """Transpose Factory object."""
        return StateSpaceInterface._transpose_factory

    @transpose_factory.setter
    def transpose_factory(value:'TransposeFactory'):
        StateSpaceInterface._transpose_factory = value

    def __init__(self):
        """ Constructor. """  
        pass

    @abc.abstractmethod
    def compute(self,u):
        """ Computes the result of a vector applied to this state space entity.

        Args:
            u: Vector which is applied.

        Returns:
            Resulting state vector x and result vector y.
        """
        pass

    @abc.abstractmethod
    def compile(self):
        """ Compiles state space entity into more basic, faster computable operations.

        Returns:
            State space entity in the form of fast computable operations.
        """
        pass 

    @abc.abstractmethod
    def realize(self):
        """ Generates a concrete realization of the state space entity.

        Returns:
            Realization of the state space entity.
        """

    @abc.abstractproperty
    def causality(self):
        """ Either causal, anticausal or mixed. """
        pass

    def add(self,rhs:'StateSpaceInterface'):
        """ Addition in state space.

        Args:
            rhs: Right hand side of the addition operation.
        
        Returns:
            Addition result in state space.
        """      
        return StateSpaceInterface.add_factory.get_add(self,rhs)
    
    def mul(self,rhs:'StateSpaceInterface'):
        """ Multiplication in state space.

        Args:
            rhs: Right hand side of the multiplication operation.
        
        Returns:
            Multiplication result in state space.
        """
        return StateSpaceInterface.multiply_factory.get_multiply(self,rhs)
    
    def neg(self):
        """ Negation in state space.
        
        Returns:
            Negation result in state space.
        """
        return StateSpaceInterface.negate_factory.get_negate(self)
    
    def inv(self):
        """ Inversion in state space.
        
        Returns:
            Inversion result in state space.
        """
        raise NotImplementedError("Not implemented yet")
    
    def transpose(self):
        """ Transposition in state space.
        
        Returns:
            Transposition result in state space.
        """
        return StateSpaceInterface.transpose_factory.get_transpose(self)

    def convert(self,into:Causality):
        """ Conversion of a state space entity to a differnt causality type.

        Args:
            into: The causality into which the state space entity shall be converted.
        
        Returns:
            Conversion result in state space.
        """
        raise NotImplementedError("Not implemented yet")
