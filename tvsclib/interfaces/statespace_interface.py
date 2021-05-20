""" Definition of the state  space interface. """
import abc
from tvsclib.causality import Causality

class StateSpaceInterface(object):
    __metaclass__ = abc.ABCMeta
    """ Classes which inherit this interface can represent
    operations and values in state space.

    Attributes:
        add_factory: Used to generate addition operations.
        negate_factory: Used to generate negation operations.
        invert_factory: Used to generate inversion operations.
        transpose_factory: Used to generate transposition operations.
    """

    add_factory = None
    negate_factory = None
    multiply_factory = None
    invert_factory = None
    transpose_factory = None
    convert_factory = None

    def __init__(self,compute_function=None):
        """ Constructor. 
        
        Args:
            compute_function: Callback to compute function. Is optional,
                              if an operation needs to be compiled before it
                              can be executed (e.g. Transpose) compute_function
                              must not be set.
        """
        self.__compute_function = compute_function

    def compile(self):
        """ Generates a state space object that has a compute function.

        Returns:
            State space object with compute function.
        """
        if self.__compute_function is None:
            raise RuntimeError("State space object without compute function has to override compile()")
        return self

    @abc.abstractmethod
    def compute(self,u):
        """ Applies a vector to this state space object.

        Args:
            u: Vector which is applied.

        Returns:
            Resulting state vector x and result vector y.
        """
        if self.__compute_function is not None:
            return self.__compute_function(u)
        return self.compile().compute(u)

    @abc.abstractmethod
    def realize(self):
        """ Generates a concrete realization.

        Returns:
            Realization object.
        """
        pass

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
        return StateSpaceInterface.invert_factory.get_invert(self)
    
    def transpose(self):
        """ Transposition in state space.
        
        Returns:
            Transposition result in state space.
        """
        return StateSpaceInterface.transpose_factory.get_transpose(self)

    def convert(self,into:Causality):
        """ Conversion of a state space object to a differnt causality type.

        Args:
            into: The causality into which the state space object shall be converted.
        
        Returns:
            Conversion result in state space.
        """
        return StateSpaceInterface.convert_factory.get_convert(self,into)
