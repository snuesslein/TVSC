""" Definition of the add mixed class. """
import numpy as np
from tvsclib.causality import Causality
from tvsclib.realization_mixed import RealizationMixed
from tvsclib.interfaces.statespace_interface import StateSpaceInterface

class AddMixed(StateSpaceInterface):
    """ Represents an addition of two mixed systems in state space. 
    
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
            raise AttributeError("AddMixed lhs_op and rhs_op have different causalities")
        if lhs_op.causality is not Causality.MIXED:
            raise AttributeError("AddMixed can not handle strict systems")
        self.lhs_op = lhs_op
        self.rhs_op = rhs_op

    def transpose(self):
        """ Override transpose method.
        Computing lhs' + rhs' is equal to (lhs + rhs)'.

        returns:
            Transposed add operation.
        """
        return AddMixed(self.lhs_op.transpose(),self.rhs_op.transpose())

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
        
        return RealizationMixed(
            causal_system = realization_lhs.causal_system.add(
                realization_rhs.causal_system).realize(),
            anticausal_system = realization_lhs.anticausal_system.add(
                realization_rhs.anticausal_system).realize())
