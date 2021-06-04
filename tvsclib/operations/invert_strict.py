""" Definition of the invert strict class. """
import numpy as np
from numpy.core.numeric import allclose
from tvsclib.causality import Causality
from tvsclib.realization_strict import RealizationStrict
from tvsclib.interfaces.statespace_interface import StateSpaceInterface

class InvertStrict(StateSpaceInterface):
    """ Represents a invert operation in state space. """

    @property
    def causality(self):
        return Causality.MIXED

    def __init__(self,operand:StateSpaceInterface):
        """ Constructor.
        
        Args:
            operand: Operand to invert.
        """
        super().__init__()
        if operand.causality is Causality.MIXED:
            raise AttributeError("InvertStrict can not handle mixed systems")
        self.operand = operand

    def transpose(self):
        """ Override transpose method.
        Computing (op^-1)' is equal to (op')^-1.

        returns:
            Transposed invert operation.
        """
        return InvertStrict(self.operand.transpose())

    def compile(self):
        """ Returns a state space operation that can be directly computed.
        Here, for the inversion, this can be expensive since it
        involves the realization of the operand.

        Returns:
            Inversion in state space.
        """
        operand_realization = self.operand.realize()
        if operand_realization.causality is Causality.CAUSAL:
            T_ol, V_r = self._rq_forward(operand_realization)
            V_l, T_o = self._ql_backward(T_ol)
            T_o_inverse = self._elementary_inversion(T_o)
            return V_r.transpose().mul(T_o_inverse).mul(V_l.transpose())
        else:
            T_ol, V_r = self._rq_forward(
                operand_realization.transpose().realize())
            V_l, T_o = self._ql_backward(T_ol)
            T_o_inverse = self._elementary_inversion(T_o)
            return V_l.mul(T_o_inverse.transpose()).mul(V_r)
    
    def realize(self):
        """ Generates a state space realization of the invert operation. 
        
        Returns:
            Realization of invert operation.
        """
        return self.compile().realize()

    def _elementary_inversion(self, realization:RealizationStrict):
        k = len(realization.A)
        A_inverse = []
        B_inverse = []
        C_inverse = []
        D_inverse = []
        for i in range(k):
            if realization.D[i].shape[0] > \
                realization.D[i].shape[1]:
                q, r = np.linalg.qr(realization.D[i])
                D_inverse.append(
                    np.linalg.inv(r) @ q.transpose())
            else:
                q, r = np.linalg.qr(
                    realization.D[i].transpose())
                D_inverse.append(
                    q @ np.linalg.inv(r).transpose())
            B_inverse.append(
                realization.B[i] @ D_inverse[i])
            C_inverse.append(
                -D_inverse[i] @ realization.C[i])
            A_inverse.append(
                realization.A[i] - B_inverse[i]\
                    @ realization.C[i])
        return RealizationStrict(causal=realization.causal,
            A=A_inverse, B=B_inverse, C=C_inverse,
            D=D_inverse)

    def _rq_forward(self, realization:RealizationStrict):
        k = len(realization.A)
        Y_matricies = [np.zeros((0,0))]
        Bo_matricies = []
        Do_matricies = []
        Av_matricies = []
        Bv_matricies = []
        Cv_matricies = []
        Dv_matricies = []

        for i in range(k):
            X_matrix = np.vstack([
                np.hstack([
                    realization.A[i] @ Y_matricies[i],
                    realization.B[i]]),
                np.hstack([
                    realization.C[i] @ Y_matricies[i],
                    realization.D[i]])])
            # Econ RQ-Decomposition
            X_matrix = X_matrix[
                range(X_matrix.shape[0]-1,-1,-1),:]
            Q_matrix, R_matrix = np.linalg.qr(
                X_matrix.transpose(), mode='reduced')
            Q_matrix = Q_matrix.transpose()
            Q_matrix = Q_matrix[
                range(Q_matrix.shape[0]-1,-1,-1),:]
            R_matrix = R_matrix.transpose()
            R_matrix = R_matrix[
                range(R_matrix.shape[0]-1,-1,-1),:]
            R_matrix = R_matrix[
                :,range(R_matrix.shape[1]-1,-1,-1)]
            
            no_rows_Y = R_matrix.shape[0] - realization.D[i].shape[0]
            no_cols_Y = R_matrix.shape[1] - realization.D[i].shape[0]
            no_cols_Y = max(0, no_cols_Y)
            Y_matricies.append(R_matrix[0:no_rows_Y,:][:,0:no_cols_Y])

            Bo_matricies.append(R_matrix[0:no_rows_Y,:][:,no_cols_Y:])
            Do_matricies.append(R_matrix[no_rows_Y:,:][:,no_cols_Y:])

            Dv_matricies.append(Q_matrix[
                Q_matrix.shape[0]-Do_matricies[i].shape[1]:,:][
                    :,Q_matrix.shape[1]-realization.D[i].shape[1]:])
            Bv_matricies.append(Q_matrix[
                0:Q_matrix.shape[0]-Do_matricies[i].shape[1],:][
                    :,Q_matrix.shape[1]-realization.D[i].shape[1]:])
            Cv_matricies.append(Q_matrix[
                Q_matrix.shape[0]-Do_matricies[i].shape[1]:,:][
                    :,0:Q_matrix.shape[1]-realization.D[i].shape[1]])
            Av_matricies.append(Q_matrix[
                0:Q_matrix.shape[0]-Do_matricies[i].shape[1],:][
                    :,0:Q_matrix.shape[1]-realization.D[i].shape[1]])
        return (
            RealizationStrict(
                causal=True,
                A=realization.A,
                B=Bo_matricies,
                C=realization.C,
                D=Do_matricies),
            RealizationStrict(
                causal=True,
                A=Av_matricies,
                B=Bv_matricies,
                C=Cv_matricies,
                D=Dv_matricies))

    def _ql_backward(self, realization:RealizationStrict):
        k = len(realization.A)
        Y_matricies = [np.zeros((0,0))] * (k+1)
        Cl_matricies = [np.zeros((0,0))] * k
        Dl_matricies = [np.zeros((0,0))] * k
        Aq_matricies = [np.zeros((0,0))] * k
        Bq_matricies = [np.zeros((0,0))] * k
        Cq_matricies = [np.zeros((0,0))] * k
        Dq_matricies = [np.zeros((0,0))] * k

        for i in range(k-1,-1,-1):
            X_matrix = np.vstack([
                np.hstack([
                    Y_matricies[i+1] @ realization.A[i],
                    Y_matricies[i+1] @ realization.B[i]
                ]),
                np.hstack([
                    realization.C[i],
                    realization.D[i]])])
            # Econ QL-Decomposition
            X_matrix = X_matrix.transpose()
            X_matrix = X_matrix[
                range(X_matrix.shape[0]-1,-1,-1),:]
            Q_matrix, L_matrix = np.linalg.qr(
                X_matrix.transpose(), mode='reduced')
            Q_matrix = Q_matrix[
                :,range(Q_matrix.shape[1]-1,-1,-1)]
            L_matrix = L_matrix[
                range(L_matrix.shape[0]-1,-1,-1),:]
            L_matrix = L_matrix[
                :,range(L_matrix.shape[1]-1,-1,-1)]
            
            no_rows_Y = L_matrix.shape[0] - realization.D[i].shape[1]
            no_rows_Y = max(0, no_rows_Y)
            no_cols_Y = realization.A[i].shape[1]

            Y_matricies[i] = L_matrix[0:no_rows_Y,:][:,0:no_cols_Y]

            Cl_matricies[i] = L_matrix[no_rows_Y:,:][:,0:no_cols_Y]
            Dl_matricies[i] = L_matrix[no_rows_Y:,:][:,no_cols_Y:]

            Dq_matricies[i] = Q_matrix[
                Q_matrix.shape[0]-realization.D[i].shape[0]:,:][
                    :,Q_matrix.shape[1]-Dl_matricies[i].shape[0]:]
            Bq_matricies[i] = Q_matrix[
                0:Q_matrix.shape[0]-realization.D[i].shape[0],:][
                    :,Q_matrix.shape[1]-Dl_matricies[i].shape[0]:]
            Cq_matricies[i] = Q_matrix[
                Q_matrix.shape[0]-realization.D[i].shape[0]:,:][
                    :,0:Q_matrix.shape[1]-Dl_matricies[i].shape[0]]
            Aq_matricies[i] = Q_matrix[
                0:Q_matrix.shape[0]-realization.D[i].shape[0],:][
                    :,0:Q_matrix.shape[1]-Dl_matricies[i].shape[0]]
        return (
            RealizationStrict(
                causal=True,
                A=Aq_matricies,
                B=Bq_matricies,
                C=Cq_matricies,
                D=Dq_matricies),
            RealizationStrict(
                causal=True,
                A=realization.A,
                B=realization.B,
                C=Cl_matricies,
                D=Dl_matricies))