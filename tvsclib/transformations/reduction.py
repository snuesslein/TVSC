import numpy as np
from typing import List
from tvsclib.stage import Stage
from tvsclib.transformation import Transformation

class Reduction(Transformation):
    def __init__(self, epsilon:float = 1e-10):
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

        if True:# not self.inplace
            stages = [Stage(stage.A_matrix.copy(),stage.B_matrix.copy(),\
            stage.C_matrix.copy(),stage.D_matrix.copy()) for stage in stages]


        k = len(stages)
        # Step 1: Reduction to a reachable system
        for i in range(k-1):
            U,s,Vt= np.linalg.svd(np.hstack([stages[i].A_matrix,stages[i].B_matrix]))
            n = np.count_nonzero(s>self.epsilon)

            rs = np.sqrt(s[:n])
            Us=U[:,:n]*rs
            sVt=rs.reshape(-1,1)*Vt[:n,:]

            stages[i].A_matrix=sVt[:,:stages[i].A_matrix.shape[1]]
            stages[i].B_matrix=sVt[:,stages[i].A_matrix.shape[1]:]
            stages[i+1].A_matrix = stages[i+1].A_matrix@Us
            stages[i+1].C_matrix = stages[i+1].C_matrix@Us

        # Step 2: Reduction to an observable system
        for i in range(k-1, 0,-1):
            U,s,Vt= np.linalg.svd(np.vstack([stages[i].C_matrix,stages[i].A_matrix]))
            n = np.count_nonzero(s>self.epsilon)

            rs = np.sqrt(s[:n])
            Us=U[:,:n]*rs
            sVt=rs.reshape(-1,1)*Vt[:n,:]

            stages[i].C_matrix=Us[:stages[i].C_matrix.shape[0],:]
            stages[i].A_matrix=Us[stages[i].C_matrix.shape[0]:,:]
            stages[i-1].A_matrix=sVt@stages[i-1].A_matrix
            stages[i-1].B_matrix=sVt@stages[i-1].B_matrix
        return stages

    def _transform_anticausal(self, stages:List[Stage]) -> List[Stage]:
        """_transform_anticausal Transforms anticausal stages to reduced form
        Args:
            stages (List[Stage]): Anticausal stages
        Returns:
            List[Stage]: Transformed anticausal stages
        """
        if True:# not self.inplace
            stages = [Stage(stage.A_matrix.copy(),stage.B_matrix.copy(),\
            stage.C_matrix.copy(),stage.D_matrix.copy()) for stage in stages]


        k = len(stages)
        # Step 1: Reduction to an observable system
        for i in range(k-1):
            U,s,Vt= np.linalg.svd(np.vstack([stages[i].C_matrix,stages[i].A_matrix]))
            n = np.count_nonzero(s>self.epsilon)

            rs = np.sqrt(s[:n])
            Us=U[:,:n]*rs
            sVt=rs.reshape(-1,1)*Vt[:n,:]

            stages[i].C_matrix=Us[:stages[i].C_matrix.shape[0],:]
            stages[i].A_matrix=Us[stages[i].C_matrix.shape[0]:,:]
            stages[i+1].A_matrix=sVt@stages[i+1].A_matrix
            stages[i+1].B_matrix=sVt@stages[i+1].B_matrix
        # Step 2: Reduction to a reachable system
        for i in range(k-1, 0,-1):
            U,s,Vt= np.linalg.svd(np.hstack([stages[i].A_matrix,stages[i].B_matrix]))
            n = np.count_nonzero(s>self.epsilon)

            rs = np.sqrt(s[:n])
            Us=U[:,:n]*rs
            sVt=rs.reshape(-1,1)*Vt[:n,:]

            stages[i].A_matrix=sVt[:,:stages[i].A_matrix.shape[1]]
            stages[i].B_matrix=sVt[:,stages[i].A_matrix.shape[1]:]
            stages[i-1].A_matrix = stages[i-1].A_matrix@Us
            stages[i-1].C_matrix = stages[i-1].C_matrix@Us


        return stages
