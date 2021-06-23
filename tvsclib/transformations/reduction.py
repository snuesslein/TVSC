import numpy as np
from typing import List
from tvsclib.stage import Stage
from tvsclib.transformation import Transformation

class Reduction(Transformation):
    def __init__(self, epsilon:float = 1e-15):
        """__init__ Constructor for state reduction transformation
        """
        super().__init__("state-reduction", self._transform_causal, self._transform_anticausal)
        self.epsilon = epsilon
    
    def _transform_causal(self, stages:List[Stage]) -> List[Stage]:
        """_transform_causal Transforms causal stages to reduced (fully reachable and observable) form

        Args:
            stages (List[Stage]): Causal stages

        Returns:
            List[Stage]: Transformed causal stages
        """
        k = len(stages)
        # Step 1: Reduction to a reachable system
        s_matricies = [np.zeros((0,0))] * (k+1)
        result_reachable:List[Stage] = []
        for i in range(k):
            X_matrix = np.hstack([
                stages[i].A_matrix @ s_matricies[i],
                stages[i].B_matrix
            ])
            _,r = np.linalg.qr(X_matrix.transpose(), 'complete')
            s_matricies[i+1] = r[0:sum(np.linalg.norm(r,axis=1) > self.epsilon),:].transpose()
            s_inv = np.linalg.inv(s_matricies[i+1].transpose() @ s_matricies[i+1]) @ s_matricies[i+1].transpose()
            A_matrix = s_inv @ stages[i].A_matrix @ s_matricies[i]
            B_matrix = s_inv @ stages[i].B_matrix
            C_matrix = stages[i].C_matrix @ s_matricies[i]
            result_reachable.append(Stage(
                A_matrix.copy(),
                B_matrix.copy(),
                C_matrix.copy(),
                stages[i].D_matrix.copy()))
        # Step 2: Reduction to an observable system
        t_matricies = [np.zeros((0,0))] * (k+1)
        result_observable:List[Stage] = []
        for i in range(k-1,-1,-1):
            X_matrix = np.vstack([
                t_matricies[i+1] @ result_reachable[i].A_matrix,
                result_reachable[i].C_matrix
            ])
            _,r = np.linalg.qr(X_matrix,'complete')
            t_matricies[i] = r[0:sum(np.linalg.norm(r,axis=1) > self.epsilon),:]
            t_inv = t_matricies[i].transpose() @ np.linalg.inv(t_matricies[i] @ t_matricies[i].transpose())
            A_matrix = t_matricies[i+1] @ result_reachable[i].A_matrix @ t_inv
            B_matrix = t_matricies[i+1] @ result_reachable[i].B_matrix
            C_matrix = result_reachable[i].C_matrix @ t_inv
            result_observable.append(Stage(
                A_matrix.copy(),
                B_matrix.copy(),
                C_matrix.copy(),
                result_reachable[i].D_matrix.copy()))
        result_observable.reverse()
        return result_observable

    def _transform_anticausal(self, stages:List[Stage]) -> List[Stage]:
        """_transform_anticausal Transforms anticausal stages to reduced form

        Args:
            stages (List[Stage]): Anticausal stages

        Returns:
            List[Stage]: Transformed anticausal stages
        """
        k = len(stages)
        # Step 1: Reduction to a reachable system
        s_matricies = [np.zeros((0,0))] * (k+1)
        result_reachable:List[Stage] = []
        for i in range(k-1,-1,-1):
            X_matrix = np.vstack([
                s_matricies[i+1].transpose() @ stages[i].A_matrix.transpose(),
                stages[i].B_matrix.transpose()
            ])
            _,r = np.linalg.qr(X_matrix,'complete')
            s_matricies[i] = r[0:sum(np.linalg.norm(r,axis=1) > 0),:].transpose()
            s_inv = np.linalg.inv(s_matricies[i].transpose() @ s_matricies[i]) @ s_matricies[i].transpose()
            A_matrix = s_inv @ stages[i].A_matrix @ s_matricies[i+1]
            B_matrix = s_inv @ stages[i].B_matrix
            C_matrix = stages[i].C_matrix @ s_matricies[i+1]
            result_reachable.append(Stage(
                A_matrix.copy(),
                B_matrix.copy(),
                C_matrix.copy(),
                stages[i].D_matrix.copy()))
        result_reachable.reverse()
        # Step 2: Reduction to an observable system
        t_matricies = [np.zeros((0,0))] * (k+1)
        result_observable:List[Stage] = []
        for i in range(k):
            X_matrix = np.vstack([
                t_matricies[i] @ result_reachable[i].A_matrix,
                result_reachable[i].C_matrix
            ])
            _,r = np.linalg.qr(X_matrix,'complete')
            t_matricies[i+1] = r[0:sum(np.linalg.norm(r,axis=1) > 0),:]
            t_inv = t_matricies[i+1].transpose() @ np.linalg.inv(t_matricies[i+1] @ t_matricies[i+1].transpose())
            A_matrix = t_matricies[i] @ result_reachable[i].A_matrix @ t_inv
            B_matrix = t_matricies[i] @ result_reachable[i].B_matrix
            C_matrix = result_reachable[i].C_matrix @ t_inv
            result_observable.append(Stage(
                A_matrix.copy(),
                B_matrix.copy(),
                C_matrix.copy(),
                result_reachable[i].D_matrix.copy()))
        return result_observable