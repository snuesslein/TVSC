import numpy as np
from typing import List, Tuple
from scipy.linalg import block_diag
from tvsclib.stage import Stage
from tvsclib.strict_system import StrictSystem
from tvsclib.mixed_system import MixedSystem
from tvsclib.expressions.multiply import Multiply
from tvsclib.expressions.const import Const
from tvsclib.expressions.strict.transpose import transpose as transposeStrict
from tvsclib.expressions.mixed.transpose import transpose as transposeMixed

def invert(system:MixedSystem) -> Multiply:
    """invert Inversion in state space

    Args:
        system (MixedSystem): System to invert

    Returns:
        Multiply: Expression which computes inverese
    """
    if np.sum(system.dims_out) >= np.sum(system.dims_in):
        # Tall matrix
        (system_V, system_v, system_R_inverted, system_U_transposed) = _inverse_factors(system)
        const_V_transpose = Const(transposeStrict(system_V), "V'")
        const_v = Const(system_v, "v")
        const_r_inverted = Const(system_R_inverted, "R^-1")
        const_U_transpose = Const(system_U_transposed, "U'")
        result = Multiply(const_V_transpose, const_v)
        result = Multiply(result, const_r_inverted)
        return Multiply(result, const_U_transpose)
    else:
        # Wide matrix
        (system_V, system_v, system_R_inverted, system_U_transposed) = _inverse_factors(
            transposeMixed(system))
        const_U = Const(transposeStrict(system_U_transposed), "U")
        const_r_inverted_transpose = Const(transposeStrict(system_R_inverted), "R'^-1")
        const_v_transpose = Const(transposeStrict(system_v), "v'")
        const_V = Const(system_V, "V")
        result = Multiply(const_U, const_r_inverted_transpose)
        result = Multiply(result, const_v_transpose)
        return Multiply(result, const_V)


def _inverse_factors(system:MixedSystem) -> Tuple[StrictSystem, StrictSystem, StrictSystem, StrictSystem]:
    causal_system = system.causal_system.copy()
    anticausal_system = system.anticausal_system.copy()

    k = len(causal_system.stages)
    # Move pass-through parts into causal system
    for i in range(k):
        causal_system.stages[i].D_matrix = causal_system.stages[i].D_matrix \
            + anticausal_system.stages[i].D_matrix
        anticausal_system.stages[i].D_matrix = \
            np.zeros(anticausal_system.stages[i].D_matrix.shape)

    # Phase 1: conversion to upper
    stages_v: List[Stage] = [] 
    d_matricies: List[np.ndarray] = []
    G_matricies: List[np.ndarray] = []
    y_matricies: List[np.ndarray] = [np.zeros((0,0))]
    for i in range(k):
        X_matrix = np.vstack([
            np.hstack([
                causal_system.stages[i].C_matrix @ y_matricies[i],
                causal_system.stages[i].D_matrix
            ]),
            np.hstack([
                causal_system.stages[i].A_matrix @ y_matricies[i],
                causal_system.stages[i].B_matrix
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
        
        no_rows_y = causal_system.stages[i].B_matrix.shape[0]
        no_cols_y = min(no_rows_y, R_matrix.shape[1])
        y_matricies.append(R_matrix[R_matrix.shape[0]-no_rows_y:,:][
            :,R_matrix.shape[1]-no_cols_y:])
                
        G_matricies.append(R_matrix[
            0:R_matrix.shape[0]-no_rows_y,:][
                :,R_matrix.shape[1]-no_cols_y:])
        d_matricies.append(
            R_matrix[0:R_matrix.shape[0]-no_rows_y,:][
                :,0:R_matrix.shape[1]-no_cols_y])
        
        stages_v.append(Stage(
            Q_matrix[d_matricies[i].shape[1]:][
                :,0:y_matricies[i].shape[1]],
            Q_matrix[d_matricies[i].shape[1]:][
                :,y_matricies[i].shape[1]:],
            Q_matrix[0:d_matricies[i].shape[1],:][
                :,0:y_matricies[i].shape[1]],
            Q_matrix[0:d_matricies[i].shape[1],:][
                :,y_matricies[i].shape[1]:]))
    
    b_matricies: List[np.ndarray] = []
    h_matricies: List[np.ndarray] = []
    g_matricies: List[np.ndarray] = []
    for i in range(k):
        b_matricies.append(block_diag(
            anticausal_system.stages[i].A_matrix, stages_v[i].A_matrix.transpose()))
        b_matricies[i][0:anticausal_system.stages[i].A_matrix.shape[0],:][
            :,anticausal_system.stages[i].A_matrix.shape[1]:
            ] = anticausal_system.stages[i].B_matrix @ stages_v[i].B_matrix.transpose()
        h_matricies.append(np.vstack([
            anticausal_system.stages[i].B_matrix @ stages_v[i].D_matrix.transpose(),
            stages_v[i].C_matrix.transpose()]))
        g_matricies.append(np.hstack([
            anticausal_system.stages[i].C_matrix,
            G_matricies[i]]))
    
    system_V = StrictSystem(causal=True, stages=stages_v)

    # Phase 2: computing the kernel and the co-range
    y_matricies = [np.zeros((0,0))] * (k+1)
    stages_v: List[Stage] = []
    stages_o: List[Stage] = []

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

        stages_o.append(Stage(
            b_matricies[i].transpose(),
            g_matricies[i].transpose(),
            R_matrix[0:no_rows_y,:][
                :,no_cols_y:].transpose(),
            R_matrix[no_rows_y:,:][
                :,no_cols_y:].transpose()
        ))

        Q_matrix = Q_matrix.transpose()
        stages_v.append(Stage(
            Q_matrix[0:y_matricies[i+1].shape[1],:][
                :,0:no_cols_y],
            Q_matrix[0:y_matricies[i+1].shape[1],:][
                :,no_cols_y:],
            Q_matrix[y_matricies[i+1].shape[1]:,:][
                :,0:no_cols_y],
            Q_matrix[y_matricies[i+1].shape[1]:,:][
                :,no_cols_y:]
        ))

    stages_o.reverse()
    stages_v.reverse()
    system_v = StrictSystem(causal=True, stages=stages_v)

    # Phase 3: computing the range, the co-kernel and R
    Y_matricies = [np.zeros((0,0))]
    stages_r: List[Stage] = []
    stages_u: List[Stage] = []

    for i in range(k):
        X_matrix = np.vstack([
            np.hstack([
                stages_o[i].A_matrix @ Y_matricies[i],
                stages_o[i].B_matrix
            ]),
            np.hstack([
                stages_o[i].C_matrix @ Y_matricies[i],
                stages_o[i].D_matrix
            ])
        ])
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
        
        no_rows_Do = stages_o[i].D_matrix.shape[0]
        no_cols_Do = no_rows_Do

        stages_r.append(Stage(
            stages_o[i].A_matrix.transpose(),
            stages_o[i].C_matrix.transpose(),
            R_matrix[0:R_matrix.shape[0]-no_rows_Do,:][
                :,R_matrix.shape[1]-no_cols_Do:].transpose(),
            R_matrix[R_matrix.shape[0]-no_rows_Do:,:][
                :,R_matrix.shape[1]-no_cols_Do:].transpose()
        ))
        Y_matricies.append(R_matrix[0:R_matrix.shape[0]-no_rows_Do,:][
            :,0:R_matrix.shape[1]-no_cols_Do])
        
        no_rows_Du = stages_r[i].D_matrix.shape[1]
        no_cols_Du = stages_o[i].D_matrix.shape[1]
        stages_u.append(Stage(
            Q_matrix[0:Q_matrix.shape[0]-no_rows_Du,:][
                :,0:Q_matrix.shape[1]-no_cols_Du],
            Q_matrix[0:Q_matrix.shape[0]-no_rows_Du,:][
                :,Q_matrix.shape[1]-no_cols_Du:],
            Q_matrix[Q_matrix.shape[0]-no_rows_Du:,:][
                :,0:Q_matrix.shape[1]-no_cols_Du],
            Q_matrix[Q_matrix.shape[0]-no_rows_Du:,:][
                :,Q_matrix.shape[1]-no_cols_Du:]
        ))

    system_U_transposed = StrictSystem(causal=True, stages=stages_u)

    # Invert factor R in state space via arrow reversal
    stages_r_inverted: List[Stage] = []
    for i in range(k):
        D_matrix = np.linalg.inv(stages_r[i].D_matrix)
        B_matrix = stages_r[i].B_matrix @ D_matrix
        C_matrix = -D_matrix @ stages_r[i].C_matrix
        A_matrix = stages_r[i].A_matrix - B_matrix @ stages_r[i].C_matrix
        stages_r_inverted.append(Stage(A_matrix, B_matrix, C_matrix, D_matrix))
    
    system_R_inverted = StrictSystem(causal=False, stages=stages_r_inverted)

    # Return factors
    return (system_V, system_v, system_R_inverted, system_U_transposed)