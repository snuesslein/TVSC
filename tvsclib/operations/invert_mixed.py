""" Definition of the invert mixed class. """
import numpy as np
from scipy.linalg import block_diag
from tvsclib.causality import Causality
from tvsclib.realization_strict import RealizationStrict
from tvsclib.realization_mixed import RealizationMixed
from tvsclib.interfaces.statespace_interface import StateSpaceInterface

class InvertMixed(StateSpaceInterface):
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
        if operand.causality is not Causality.MIXED:
            raise AttributeError("InvertMixed can not handle strict systems")
        self.operand = operand

    def transpose(self):
        """ Override transpose method.
        Computing (op^-1)' is equal to (op')^-1.

        returns:
            Transposed invert operation.
        """
        return InvertMixed(self.operand)

    def compile(self):
        """ Returns a state space operation that can be directly computed.
        Here, for the inversion, this can be expensive since it
        involves the realization of the operand.

        Returns:
            Inversion in state space.
        """
        operand_realization = self.operand.realize()
        if np.sum(operand_realization.dims_out) \
            >= np.sum(operand_realization.dims_in):
            return self._left_inverse(operand_realization)
        else:
            return self._left_inverse(
                operand_realization.transpose().realize()).transpose()

    def realize(self):
        """ Generates a state space realization of the invert operation. 
        
        Returns:
            Realization of invert operation.
        """
        return self.compile().realize()

    def _left_inverse(self, realization: RealizationMixed):
        causal_system = realization.causal_system
        anticausal_system = realization.anticausal_system
        k = len(causal_system.A)
        # Move pass-through parts into causal system
        for i in range(k):
            causal_system.D[i] = causal_system.D[i] + anticausal_system.D[i]
            anticausal_system.D[i] = np.zeros(anticausal_system.D[i].shape)
        # Phase 1: conversion to upper
        av_matricies = []
        qv_matricies = []
        pv_matricies = []
        dv_matricies = []
        d_matricies = []
        G_matricies = []
        y_matricies = [np.zeros((0,0))] * (k+1)
        for i in range(k):
            X_matrix = np.vstack([
                np.hstack([
                    causal_system.C[i] @ y_matricies[i],
                    causal_system.D[i]
                ]),
                np.hstack([
                    causal_system.A[i] @ y_matricies[i],
                    causal_system.B[i]
                ])
            ])
            # RQ-Decomposition
            X_matrix = X_matrix[
                range(X_matrix.shape[0]-1,-1,-1),:]
            Q_matrix, R_matrix = np.linalg.qr(
                X_matrix.transpose(), mode='complete')
            Q_matrix = Q_matrix.transpose()
            Q_matrix = Q_matrix[
                range(Q_matrix.shape[0]-1,-1,-1),:]
            R_matrix = R_matrix.transpose()
            R_matrix = R_matrix[
                range(R_matrix.shape[0]-1,-1,-1),:]
            R_matrix = R_matrix[
                :,range(R_matrix.shape[1]-1,-1,-1)]

            no_rows_y = causal_system.B[i].shape[0]
            no_cols_y = min(no_rows_y, R_matrix.shape[1])
            y_matricies[i+1] = R_matrix[R_matrix.shape[0]-no_rows_y:,:][
                :,R_matrix.shape[1]-no_cols_y:]
            
            G_matricies.append(R_matrix[
                0:R_matrix.shape[0]-no_rows_y,:][
                    :,R_matrix.shape[1]-no_cols_y:])
            d_matricies.append(
                R_matrix[0:R_matrix.shape[0]-no_rows_y,:][
                    :,0:R_matrix.shape[1]-no_cols_y])
            
            pv_matricies.append(
                Q_matrix[0:d_matricies[i].shape[1],:][
                    :,0:y_matricies[i].shape[1]])
            dv_matricies.append(
                Q_matrix[0:d_matricies[i].shape[1],:][
                    :,y_matricies[i].shape[1]:])
            av_matricies.append(
                Q_matrix[d_matricies[i].shape[1]:][
                    :,0:y_matricies[i].shape[1]])
            qv_matricies.append(
                Q_matrix[d_matricies[i].shape[1]:][
                    :,y_matricies[i].shape[1]:])
        
        b_matricies = []
        h_matricies = []
        g_matricies = []
        for i in range(k):
            b_matricies.append(block_diag(
                anticausal_system.A[i], av_matricies[i].transpose()))
            b_matricies[i][0:anticausal_system.A[i].shape[0],:][
                :,anticausal_system.A[i].shape[1]:
                ] = anticausal_system.B[i] @ qv_matricies[i].transpose()
            h_matricies.append(np.vstack([
                anticausal_system.B[i] @ dv_matricies[i].transpose(),
                pv_matricies[i].transpose()]))
            g_matricies.append(np.hstack([
                anticausal_system.C[i],
                G_matricies[i]]))

        V_realization = RealizationStrict(causal=True,
            A=av_matricies, B=qv_matricies, C=pv_matricies,
            D=dv_matricies)
        
        # Phase 2: computing the kernel and the co-range
        y_matricies = [np.zeros((0,0))] * (k+1)
        av_matricies = [np.zeros((0,0))] * k
        qv_matricies = [np.zeros((0,0))] * k
        pv_matricies = [np.zeros((0,0))] * k
        dv_matricies = [np.zeros((0,0))] * k

        bo_matricies = [np.zeros((0,0))] * k
        go_matricies = [np.zeros((0,0))] * k
        ho_matricies = [np.zeros((0,0))] * k
        do_matricies = [np.zeros((0,0))] * k

        for i in range(k-1,-1,-1):
            X_matrix = np.vstack([
                np.hstack([
                    b_matricies[i] @ y_matricies[i+1],
                    h_matricies[i]]),
                np.hstack([
                    g_matricies[i] @ y_matricies[i+1],
                    d_matricies[i]])])
            # RQ-Decomposition
            X_matrix = X_matrix[
                range(X_matrix.shape[0]-1,-1,-1),:]
            Q_matrix, R_matrix = np.linalg.qr(
                X_matrix.transpose(), mode='complete')
            Q_matrix = Q_matrix.transpose()
            Q_matrix = Q_matrix[
                range(Q_matrix.shape[0]-1,-1,-1),:]
            R_matrix = R_matrix.transpose()
            R_matrix = R_matrix[
                range(R_matrix.shape[0]-1,-1,-1),:]
            R_matrix = R_matrix[
                :,range(R_matrix.shape[1]-1,-1,-1)]
            
            no_rows_y = R_matrix.shape[0] - d_matricies[i].shape[0]
            no_cols_y = R_matrix.shape[1] - d_matricies[i].shape[0]
            no_cols_y = max(no_cols_y, 0)
            y_matricies[i] = R_matrix[0:no_rows_y,:][:,0:no_cols_y]

            bo_matricies[i] = b_matricies[i].transpose()
            ho_matricies[i] = g_matricies[i].transpose()
            go_matricies[i] = R_matrix[0:no_rows_y,:][
                :,no_cols_y:].transpose()
            do_matricies[i] = R_matrix[no_rows_y:,:][
                :,no_cols_y:].transpose()
            
            Q_matrix = Q_matrix.transpose()
            av_matricies[i] = Q_matrix[0:y_matricies[i+1].shape[1],:][
                :,0:no_cols_y]
            qv_matricies[i] = Q_matrix[0:y_matricies[i+1].shape[1],:][
                :,no_cols_y:]
            pv_matricies[i] = Q_matrix[y_matricies[i+1].shape[1]:,:][
                :,0:no_cols_y]
            dv_matricies[i] = Q_matrix[y_matricies[i+1].shape[1]:,:][
                :,no_cols_y:]
            
        v_realization = RealizationStrict(causal=True, A=av_matricies,
            B=qv_matricies, C=pv_matricies, D=dv_matricies)
        
        # Phase 3: computing the range, the co-kernel and R
        Y_matricies = [np.zeros((0,0))] * (k+1)
        Ao_matricies = []
        Bo_matricies = []
        Co_matricies = []
        Do_matricies = []

        Au_matricies = []
        Bu_matricies = []
        Cu_matricies = []
        Du_matricies = []

        for i in range(k):
            X_matrix = np.vstack([
                np.hstack([
                    bo_matricies[i] @ Y_matricies[i],
                    ho_matricies[i]]),
                np.hstack([
                    go_matricies[i] @ Y_matricies[i],
                    do_matricies[i]])])
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
            
            no_rows_Do = do_matricies[i].shape[0]
            no_cols_Do = no_rows_Do
            Do_matricies.append(R_matrix[R_matrix.shape[0]-no_rows_Do:,:][
                :,R_matrix.shape[1]-no_cols_Do:].transpose())
            Co_matricies.append(R_matrix[0:R_matrix.shape[0]-no_rows_Do,:][
                :,R_matrix.shape[1]-no_cols_Do:].transpose())
            Ao_matricies.append(bo_matricies[i].transpose())
            Bo_matricies.append(go_matricies[i].transpose())
            Y_matricies[i+1] = R_matrix[0:R_matrix.shape[0]-no_rows_Do,:][
                :,0:R_matrix.shape[1]-no_cols_Do]
            
            no_rows_Du = Do_matricies[i].shape[1]
            no_cols_Du = do_matricies[i].shape[1]
            Du_matricies.append(Q_matrix[Q_matrix.shape[0]-no_rows_Du:,:][
                :,Q_matrix.shape[1]-no_cols_Du:])
            Bu_matricies.append(Q_matrix[0:Q_matrix.shape[0]-no_rows_Du,:][
                :,Q_matrix.shape[1]-no_cols_Du:])
            Cu_matricies.append(Q_matrix[Q_matrix.shape[0]-no_rows_Du:,:][
                :,0:Q_matrix.shape[1]-no_cols_Du])
            Au_matricies.append(Q_matrix[0:Q_matrix.shape[0]-no_rows_Du,:][
                :,0:Q_matrix.shape[1]-no_cols_Du])
        
        U_transposed_realization = RealizationStrict(causal=True,
            A=Au_matricies, B=Bu_matricies, C=Cu_matricies, D=Du_matricies)
        
        # Invert factor R in state-space
        Ari_matricies = []
        Bri_matricies = []
        Cri_matricies = []
        Dri_matricies = []
        for i in range(k):
            Dri_matricies.append(np.linalg.inv(Do_matricies[i]))
            Bri_matricies.append(Bo_matricies[i] @ Dri_matricies[i])
            Cri_matricies.append(-Dri_matricies[i] @ Co_matricies[i])
            Ari_matricies.append(Ao_matricies[i]\
                - Bri_matricies[i] @ Co_matricies[i])
        
        R_inverted_realization = RealizationStrict(causal=False,
            A=Ari_matricies, B=Bri_matricies, C=Cri_matricies, D=Dri_matricies)
        
        # Assemble pseudoinverse
        return V_realization.transpose().mul(v_realization)\
            .mul(R_inverted_realization).mul(U_transposed_realization)

        

            