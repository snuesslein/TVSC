""" Definition of the add strict class. """
import numpy as np
from scipy.linalg import block_diag
from tvsclib.causality import Causality
from tvsclib.realization_strict import RealizationStrict
from tvsclib.interfaces.statespace_interface import StateSpaceInterface

class AddStrict(StateSpaceInterface):
    """ Represents an addition of two strict systems in state space. 
    
    Attributes:
        lhs_op: Left hand side operand.
        rhs_op: Right hand side operand.
    """

    @property
    def causality(self):
        return self.lhs_op.causality

    def __init__(self,lhs_op:StateSpaceInterface,rhs_op:StateSpaceInterface):
        """ Constructor.
        
        Args:
            lhs_op: Left hand side operand.
            rhs_op: Right hand side operand.
        """
        super().__init__(self._compute_function)
        if lhs_op.causality is not rhs_op.causality:
            raise AttributeError("AddStrict lhs_op and rhs_op have different causalities")
        if lhs_op.causality == Causality.MIXED:
            raise AttributeError("AddStrinct can not handle mixed systems")
        self.lhs_op = lhs_op
        self.rhs_op = rhs_op

    def transpose(self):
        """ Override transpose method.
        Computing lhs' + rhs' is equal to (lhs + rhs)'.

        returns:
            Transposed add operation.
        """
        return AddStrict(self.lhs_op.transpose(),self.rhs_op.transpose())

    def _compute_function(self,u):
        """ Applies a vector to addition result in state space.

        Args:
            u: Vector to be applied.

        Returns: 
            Resulting state vector x and result vector y.
        """
        x_lhs,y_lhs = self.lhs_op.compute(u)
        x_rhs,y_rhs = self.rhs_op.compute(u)
        x_result = np.vstack([
            x_lhs,
            x_rhs
        ])
        y_result = y_lhs + y_rhs
        return (x_result,y_result)

    def realize(self):
        """ Generates a state space realization of the addition operation. 
        
        Returns:
            Realization of addition operation.
        """
        realization_lhs = self.lhs_op.realize()
        realization_rhs = self.rhs_op.realize()

        if ~np.all(realization_lhs.dims_in == realization_rhs.dims_in):
            raise AttributeError("Input dimensions dont match")
        if ~np.all(realization_lhs.dims_out == realization_rhs.dims_out):
            raise AttributeError("Output dimensions dont match")

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
        return RealizationStrict(
            causal=realization_lhs.causal,
            A=result_A,
            B=result_B,
            C=result_C,
            D=result_D
        )