""" Definition of the add strict class. """
import numpy as np
from scipy.linalg import block_diag
from tvsclib.realization_strict import RealizationStrict
from tvsclib.interfaces.statespace_interface import StateSpaceInterface

class AddStrict(StateSpaceInterface):
    """ Represents an addition of two strict systems in state space. 
    
    Attributes:
        causality (enum): Type of the system, either causal or anticausal
        lhs_op (StateSpaceInterface): Left hand side operand
        rhs_op (StateSpaceInterface): Right hand side operand
    """

    @property
    def causality(self):
        return self.lhs_op.causality

    def __init__(self,lhs_op,rhs_op):
        """ Constructor.
        
        Args:
            lhs_op (StateSpaceInterface): Left hand side operand
            rhs_op (StateSpaceInterface): Right hand side operand
        """
        self.lhs_op = lhs_op
        self.rhs_op = rhs_op
    
    def transpose(self):
        """ Override transpose method.
        Computing lhs' + rhs' is equal to (lhs + rhs)'
        """
        return AddStrict(self.lhs_op.transpose(),self.rhs_op.transpose())

    def compute(self,u):
        """ Applies a vector to addition result in state space.

        Args:
            u (float[]): Vector to be applied

        Returns: 
            x,y: Resulting state vector x and result vector y
        """
        x_lhs,y_lhs = self.lhs_op.compute(u)
        x_rhs,y_rhs = self.rhs_op.compute(u)
        x_result = np.vstack([
            x_lhs,
            x_rhs
        ])
        y_result = y_lhs + y_rhs
        return y_result
    
    def compile(self):
        """ Returns a state space operation that can be directly computed.
        For addition trivial since it can already be computed

        Returns:
            AddStrict: Addition in state space
        """
        return self
    
    def realize(self):
        """ Generates a state space realization of the addition operation. 
        
        Returns:
            RealizationStrict: Realization of addition operation
        """
        realization_lhs = self.lhs_op.realize()
        realization_rhs = self.rhs_op.realize()

        assert np.all(realization_lhs.dims_in == realization_rhs.dims_in), "Input dimensions dont match"
        assert np.all(realization_lhs.dims_out == realization_rhs.dims_out), "Output dimensions dont match"
        assert realization_lhs.causality == realization_rhs.causality, "Causailties dont match"

        result_A = []
        result_B = []
        result_C = []
        result_D = []
        k = len(realization_lhs.A)
        for i in range(k):
            result_A.append(block_diag(
                realization_lhs.A[i],
                realization_rhs.A[i]
            ))
            result_B.append(np.vstack([
                realization_lhs.B[i],
                realization_rhs.B[i]
            ]))
            result_C.append(np.hstack([
                realization_lhs.C[i],
                realization_rhs.C[i]
            ]))
            result_D.append(
                realization_lhs.D[i] + realization_rhs.D[i]
            )
        #return RealizationStrict(
        #    causal=realization_lhs.causal,
        #    A=result_A,
        #    B=result_B,
        #    C=result_C,
        #    D=result_D
        #)