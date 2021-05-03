""" Definition of the realization mixed class. """
import numpy as np
from tvsclib.realization_strict import RealizationStrict
from tvsclib.transfer_operator import TransferOperator
from tvsclib.causality import Causality
from tvsclib.interfaces.realization_interface import RealizationInterface

class RealizationMixed(RealizationInterface):
    """ Represents a mixed realization in state space.

    Attributes:
        causality (enum): Type of the system, here always mixed
        causal_system (RealizationStrict): Causal sub system
        anticausal_system (RealizationStrict): Anticausal sub system
        dims_in (int[]): Input dimensions for each time step
        dims_out (int[]): Output dimensions for each time step
    """

    @property
    def causality(self):
        return Causality.MIXED

    @property
    def dims_in(self):
        return self.causal_system.dims_in

    @property
    def dims_out(self):
        return self.causal_system.dims_out

    def __init__(
        self,
        causal_system:RealizationStrict=None,
        anticausal_system:RealizationStrict=None,
        transferoperator:TransferOperator=None,
        separation=None):
        """ Constructor.

        Args:
            causal_system (RealizationStrict): Causal sub system
            anticausal_system (RealizationStrinct): Anticausal sub system
            transferoperator (TransferOperator): Transfer operator instance which shall be
                                                 used to generate realization
            separation (SeparationInterface): Separation object which shall be used to
                                              decompose transfer operator
        """
        if transferoperator is not None and separation is not None:
            self.causal_system = RealizationStrict(
                causal=True,
                transferoperator=transferoperator,
                separation=separation)
            self.anticausal_system = RealizationStrict(
                causal=False,
                transferoperator=transferoperator,
                separation=separation)
        else:
            self.causal_system = causal_system
            self.anticausal_system = anticausal_system

    def compute(self,u):
        """ Computes the result of a vector applied to this realization.
        The states of the causal and anticausal system are returned in stacked
        fashion. [x_causal,x_anticausal]'

        Args:
            u (float[]): Vector which is applied

        Returns:
            x,y: Resulting state vector x and result vector y
        """
        x_causal,y_causal = self.causal_system.compute(u)
        x_anticausal,y_anticausal = self.anticausal_system.compute(u)
        x_result = np.vstack([
            x_causal, x_anticausal
        ])
        y_result = y_causal + y_anticausal
        return (x_result,y_result)

    def compile(self):
        return self

    def realize(self):
        return RealizationMixed(
            causal_system=self.causal_system.realize(),
            anticausal_system=self.anticausal_system.realize()
        )

    def generate_transferoperator(self):
        """ Generates a transfer operator from the state space realization.

        Returns:
            TransferOperator: Transfer operator object
        """
        transferoperator_causal = self.causal_system.generate_transferoperator()
        transferoperator_anticausal = self.anticausal_system.generate_transferoperator()
        matrix = transferoperator_causal.matrix + transferoperator_anticausal.matrix
        return TransferOperator(
            matrix,
            transferoperator_causal.dims_in,
            transferoperator_causal.dims_out
        )
