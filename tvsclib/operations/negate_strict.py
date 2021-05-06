""" Definition of the negate strict class. """
import numpy as np
from scipy.linalg import block_diag
from tvsclib.realization_strict import RealizationStrict
from tvsclib.interfaces.statespace_interface import StateSpaceInterface

class NegateStrict(StateSpaceInterface):
    """ Represents a negation operation in state space. """
    
    @property
    def causality(self):
        return self.operand.causality

    def __init__(self,operand:StateSpaceInterface):
        """ Constructor.
        
        Args:
            operand: Operand to negate.
        """
        self.operand = operand
    
    def compute(self,u):
        """ Applies a vector to negation result in state space.

        Args:
            u: Vector to be applied.

        Returns: 
            Resulting state vector x and result vector y.
        """
        x,y = self.operand.compute(u)
        return (x,-y)

    def compile(self):
        """ Returns a state space operation that can be directly computed.
        For negation trivial since it can already be computed.

        Returns:
            Negation in state space.
        """
        return self

    def realize(self):
        """ Generates a state space realization of the negation operation. 
        
        Returns:
            Realization of negation operation.
        """
        realization = self.operand.realize()
        result_A = []
        result_B = []
        result_C = []
        result_D = []
        k = len(realization.A)
        for i in range(k):
            result_A.append(realization.A[i])
            result_B.append(realization.B[i])
            result_C.append(-realization.C[i])
            result_D.append(-realization.D[i])
        return RealizationStrict(
            causal=realization.causal,
            A=result_A,
            B=result_B,
            C=result_C,
            D=result_D
        )
