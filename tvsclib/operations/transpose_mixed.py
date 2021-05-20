""" Definition of the transpose mixed class. """
from tvsclib.causality import Causality
from tvsclib.realization_mixed import RealizationMixed
from tvsclib.interfaces.statespace_interface import StateSpaceInterface

class TransposeMixed(StateSpaceInterface):
    """ Represents a transposition operation in state space. """
    
    @property
    def causality(self):
        return Causality.MIXED

    def __init__(self,operand:StateSpaceInterface):
        """ Constructor.
        
        Args:
            operand: Operand to transpose.
        """
        super().__init__()
        if operand.causality is not Causality.MIXED:
            raise AttributeError("TransposeMixed can not handle strict systems")
        self.operand = operand

    def transpose(self):
        """ Override transpose method.
        Computing X'' is equal to X.

        returns:
            Transposed transpose operation.
        """
        return self.operand

    def compile(self):
        """ Returns a state space operation that can be directly computed.
        Here, for the transposition, this can be expensive since it
        involves the realization of the operand.

        Returns:
            Transposition in state space.
        """
        realization = self.operand.realize()
        return RealizationMixed(
            causal_system = realization.anticausal_system.transpose().realize(),
            anticausal_system = realization.causal_system.transpose().realize()
        )

    def realize(self):
        """ Generates a state space realization of the transposition operation. 
        
        Returns:
            Realization of transposition operation.
        """
        return self.compile().realize()
