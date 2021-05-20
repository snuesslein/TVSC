""" Definition of the transpose strict class. """
from tvsclib.causality import Causality
from tvsclib.realization_strict import RealizationStrict
from tvsclib.interfaces.statespace_interface import StateSpaceInterface

class TransposeStrict(StateSpaceInterface):
    """ Represents a transposition operation in state space. """
    
    @property
    def causality(self):
        if self.operand.causality is not Causality.CAUSAL:
            return Causality.CAUSAL
        else:
            return Causality.ANTICAUSAL

    def __init__(self,operand:StateSpaceInterface):
        """ Constructor.
        
        Args:
            operand: Operand to transpose.
        """
        super().__init__()
        if operand.causality == Causality.MIXED:
            raise AttributeError("TransposeStrict can not handle mixed systems")
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
        A_result = []
        B_result = []
        C_result = []
        D_result = []
        k = len(realization.A)
        for i in range(k):
            A_result.append(realization.A[i].transpose())
            B_result.append(realization.C[i].transpose())
            C_result.append(realization.B[i].transpose())
            D_result.append(realization.D[i].transpose())
        return RealizationStrict(
            causal=not realization.causal,
            A=A_result,
            B=B_result,
            C=C_result,
            D=D_result
        )

    def realize(self):
        """ Generates a state space realization of the transposition operation. 
        
        Returns:
            Realization of transposition operation.
        """
        return self.compile().realize()
