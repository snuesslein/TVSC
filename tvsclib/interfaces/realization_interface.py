""" Definition of the realization interface. """
import abc
import numpy as np
from tvsclib.interfaces.statespace_interface import StateSpaceInterface

class RealizationInterface(StateSpaceInterface):
    __metaclass__ = abc.ABCMeta
    """ Classes which inherit this interface represent
    concrete realizations of operations in state space.
    """

    def __init__(self):
        """ Constructor. """  
        pass
    
    @abc.abstractmethod
    def generate_transferoperator(self):
        """ Generates a transfer operator from state space realization.

        Returns:
            Corresponding transfer operator.
        """
        pass
    
    @abc.abstractmethod
    def transform(self,transformation:str,**kwargs):
        """ Apply state transformation.
        
        Args:
            transformation: Name of the transformation.
            **kwargs: Arguments for the specific transformation.

        Returns:
            A realization with the same input output behaviour but different
            state space.
        """
        pass
    
    @abc.abstractproperty
    def dims_in(self):
        """ Input dimensions at each time step. """
        pass
    
    @abc.abstractproperty
    def dims_out(self):
        """ Output dimensions at each time step. """
        pass

    @staticmethod
    def shiftoperator(N:int):
        """ Causal shift (downshift) matrix.

        Args:
            N: Number of rows and columns

        Returns:
            Shiftmatrix
        """
        return np.diag(*np.ones((1,N-1)),-1)