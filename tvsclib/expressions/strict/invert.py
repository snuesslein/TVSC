from tvsclib.expressions.mixed.transpose import transpose
import numpy as np
from typing import Tuple
from tvsclib.stage import Stage
from tvsclib.strict_system import StrictSystem
from tvsclib.mixed_system import MixedSystem
from tvsclib.expressions.utils.convert import convert
from tvsclib.expressions.mixed.multiply import multiply as multiplyMixed
from tvsclib.expressions.strict.transpose import transpose as transposeStrict

def invert(system:StrictSystem) -> MixedSystem:
    """invert Inversion in state space

    Args:
        system (StrictSystem): System to invert

    Returns:
        MixedSystem: Inversion result
    """
    if system.causal:
        T_ol, V_r = _rq_forward(system)
        V_l, T_o = _ql_backward(T_ol)
        T_o_inverse = _elementary_inversion(T_o)
        result = multiplyMixed(
            convert(transposeStrict(V_r), MixedSystem),
            convert(T_o_inverse, MixedSystem))
        result  = multiplyMixed(
            result,
            convert(transposeStrict(V_l), MixedSystem))
        return result
    T_ol, V_r = _rq_forward(transposeStrict(system))
    V_l, T_o = _ql_backward(T_ol)
    T_o_inverse = _elementary_inversion(T_o)
    result = multiplyMixed(
        convert(V_l, MixedSystem),
        convert(transposeStrict(T_o_inverse), MixedSystem))
    result = multiplyMixed(
        result,
        convert(V_r, MixedSystem))
    return result
    

def _elementary_inversion(system:StrictSystem) -> StrictSystem:
    k = len(system.stages)
    stages_inverse = []
    for i in range(k):
        inverse_D = None
        if system.stages[i].D_matrix.shape[0] > system.stages[i].D_matrix.shape[1]:
            q,r = np.linalg.qr(system.stages[i].D_matrix)
            inverse_D = np.linalg.inv(r) @ q.transpose()
        else:
            q,r = np.linalg.qr(system.stages[i].D_matrix.transpose())
            inverse_D = q @ np.linalg.inv(r).transpose()
        inverse_B = system.stages[i].B_matrix @ inverse_D
        stages_inverse.append(Stage(
            system.stages[i].A_matrix - inverse_B @ system.stages[i].C_matrix,
            inverse_B,
            -inverse_D @ system.stages[i].C_matrix,
            inverse_D))
    return StrictSystem(
        causal=system.causal,
        stages=stages_inverse)

def _rq_forward(system:StrictSystem) -> Tuple[StrictSystem, StrictSystem]:
    k = len(system.stages)
    Y_matricies = [np.zeros((0,0))]
    stages_r = []
    stages_q = []
    for i in range(k):
        X_matrix = np.vstack([
            np.hstack([
                system.stages[i].A_matrix @ Y_matricies[i],
                system.stages[i].B_matrix
            ]),
            np.hstack([
                system.stages[i].C_matrix @ Y_matricies[i],
                system.stages[i].D_matrix
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
            
        no_rows_Y = R_matrix.shape[0] - system.stages[i].D_matrix.shape[0]
        no_cols_Y = R_matrix.shape[1] - system.stages[i].D_matrix.shape[0]
        no_cols_Y = max(0, no_cols_Y)
        Y_matricies.append(R_matrix[0:no_rows_Y,:][:,0:no_cols_Y])

        Br_matrix = R_matrix[0:no_rows_Y,:][:,no_cols_Y:]
        Dr_matrix = R_matrix[no_rows_Y:,:][:,no_cols_Y:]

        Dq_matrix = Q_matrix[
            Q_matrix.shape[0]-Dr_matrix.shape[1]:,:][
                :,Q_matrix.shape[1]-system.stages[i].D_matrix.shape[1]:]
        Bq_matrix = Q_matrix[
            0:Q_matrix.shape[0]-Dr_matrix.shape[1],:][
                :,Q_matrix.shape[1]-system.stages[i].D_matrix.shape[1]:]
        Cq_matrix = Q_matrix[
            Q_matrix.shape[0]-Dr_matrix.shape[1]:,:][
                :,0:Q_matrix.shape[1]-system.stages[i].D_matrix.shape[1]]
        Aq_matrix = Q_matrix[
            0:Q_matrix.shape[0]-Dr_matrix.shape[1],:][
                :,0:Q_matrix.shape[1]-system.stages[i].D_matrix.shape[1]]
        
        stages_r.append(Stage(
            system.stages[i].A_matrix,
            Br_matrix,
            system.stages[i].C_matrix,
            Dr_matrix))
        stages_q.append(Stage(
            Aq_matrix, Bq_matrix, Cq_matrix, Dq_matrix))
    return (
        StrictSystem(causal=True,stages=stages_r),
        StrictSystem(causal=True,stages=stages_q))

def _ql_backward(system:StrictSystem) -> Tuple[StrictSystem, StrictSystem]:
    k = len(system.stages)
    Y_matricies = [np.zeros((0,0))] * (k+1)
    stages_q = []
    stages_l = []
    for i in range(k-1,-1,-1):
        X_matrix = np.vstack([
            np.hstack([
                Y_matricies[i+1] @ system.stages[i].A_matrix,
                Y_matricies[i+1] @ system.stages[i].B_matrix
            ]),
            np.hstack([
                system.stages[i].C_matrix,
                system.stages[i].D_matrix
            ])
        ])
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
        
        no_rows_Y = L_matrix.shape[0] - system.stages[i].D_matrix.shape[1]
        no_rows_Y = max(0, no_rows_Y)
        no_cols_Y = system.stages[i].A_matrix.shape[1]

        Y_matricies[i] = L_matrix[0:no_rows_Y,:][:,0:no_cols_Y]

        Cl_matrix = L_matrix[no_rows_Y:,:][:,0:no_cols_Y]
        Dl_matrix = L_matrix[no_rows_Y:,:][:,no_cols_Y:]

        Dq_matrix = Q_matrix[
            Q_matrix.shape[0]-system.stages[i].D_matrix.shape[0]:,:][
                :,Q_matrix.shape[1]-Dl_matrix.shape[0]:]
        Bq_matrix = Q_matrix[
            0:Q_matrix.shape[0]-system.stages[i].D_matrix.shape[0],:][
                :,Q_matrix.shape[1]-Dl_matrix.shape[0]:]
        Cq_matrix = Q_matrix[
            Q_matrix.shape[0]-system.stages[i].D_matrix.shape[0]:,:][
                :,0:Q_matrix.shape[1]-Dl_matrix.shape[0]]
        Aq_matrix = Q_matrix[
            0:Q_matrix.shape[0]-system.stages[i].D_matrix.shape[0],:][
                :,0:Q_matrix.shape[1]-Dl_matrix.shape[0]]
        
        stages_l.append(Stage(
            system.stages[i].A_matrix,
            system.stages[i].B_matrix,
            Cl_matrix,
            Dl_matrix))
        stages_q.append(Stage(
            Aq_matrix, Bq_matrix, Cq_matrix, Dq_matrix))
    stages_l.reverse()
    stages_q.reverse()
    return (
        StrictSystem(causal=True,stages=stages_q),
        StrictSystem(causal=True,stages=stages_l))

