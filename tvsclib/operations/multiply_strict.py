""" Definition of the multiply strict class. """
import numpy as np
from scipy.linalg import block_diag
from tvsclib.realization_strict import RealizationStrict
from tvsclib.interfaces.statespace_interface import StateSpaceInterface

class MultiplyStrict(StateSpaceInterface):
    """ Represents a multiplication operation in state space. """
    
    @property
    def causality(self):
        return self.lhs_op.causality

    def __init__(self,lhs_op:StateSpaceInterface,rhs_op:StateSpaceInterface):
        """ Constructor.
        
        Args:
            lhs_op: Left hand side operand.
            rhs_op: Right hand side operand.
        """
        self.lhs_op = lhs_op
        self.rhs_op = rhs_op

    def transpose(self):
        """ Override transpose method.
        Computing rhs' * lhs' is equal to (lhs * rhs)'.

        returns:
            Transposed multiply operation.
        """
        return MultiplyStrict(self.rhs_op.transpose(),self.lhs_op.transpose())

    def compute(self,u):
        """ Applies a vector to multiplication result in state space.

        Args:
            u: Vector to be applied.

        Returns: 
            Resulting state vector x and result vector y.
        """
        x_rhs,y_rhs = self.rhs_op.compute(u)
        x_lhs,y_lhs = self.lhs_op.compute(y_rhs)
        x_result = np.vstack([
            x_rhs,
            x_lhs
        ])
        y_result = y_lhs
        return (x_result,y_result)

    def compile(self):
        """ Returns a state space operation that can be directly computed.
        For multiplication trivial since it can already be computed.

        Returns:
            Multiplication in state space.
        """
        return self

    def realize(self):
        """ Generates a state space realization of the multiplication operation. 
        
        Returns:
            Realization of multiplication operation.
        """
        realization_lhs = self.lhs_op.realize()
        realization_rhs = self.rhs_op.realize()
        
        if ~np.all(realization_lhs.dims_in == realization_rhs.dims_out):
            raise AttributeError("Input/Output dimensions dont match")
        if realization_lhs.causality is not realization_rhs.causality:
            raise AttributeError("Causailties dont match")
        
        
        result_A = []
        result_B = []
        result_C = []
        result_D = []
        k = len(realization_lhs.A)
        for i in range(k):
            result_A.append(block_diag(
                realization_rhs.A[i],
                realization_lhs.A[i]
            ))
            result_A[i][
                -realization_lhs.B[i].shape[0]:,
                0:realization_rhs.C[i].shape[1]
            ] = realization_lhs.B[i] @ realization_rhs.C[i]
            result_B.append(np.vstack([
                realization_rhs.B[i],
                realization_lhs.B[i] @ realization_rhs.D[i]
            ]))
            result_C.append(np.hstack([
                realization_lhs.D[i] @ realization_rhs.C[i],
                realization_lhs.C[i]
            ]))
            result_D.append(
                realization_lhs.D[i] @ realization_rhs.D[i]
            )
        return RealizationStrict(
            causal=realization_lhs.causal,
            A=result_A,
            B=result_B,
            C=result_C,
            D=result_D
        )

