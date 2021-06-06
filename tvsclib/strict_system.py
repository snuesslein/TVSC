from __future__ import annotations
import numpy as np
from scipy.linalg import block_diag
from copy import deepcopy
from typing import List, Tuple
from tvsclib.stage import Stage
from tvsclib.causality import Causality
from tvsclib.system_identification_interface import SystemIdentificationInterface
from tvsclib.system_interface import SystemInterface

class StrictSystem(SystemInterface):
    def __init__(self, causal:bool, system_identification:SystemIdentificationInterface = None, stages:List[Stage] = None):
        """__init__ Constructor. Creates a strict state space system either with a given list of stages or with an system identification
        interface.

        Args:
            causal (bool): If true a causal system is created, otherwise an anticausal system is created.
            system_identification (SystemIdentificationInterface, optional): System identification object. Defaults to None.
            stages (List[Stage], optional): List of stages which define the strict state space system. Defaults to None.

        Raises:
            AttributeError: Raises if not enough arguments are given.
        """
        self.causal = causal
        if system_identification is not None:
            self.stages = system_identification.get_stages(causal)
        elif stages is not None:
            self.stages = stages
        else:
            raise AttributeError("Not enough arguments provided")
    
    def copy(self) -> StrictSystem:
        """copy Returns a copy of this system

        Returns:
            StrictSystem: Copy of this system
        """
        return StrictSystem(causal=self.causal, stages=deepcopy(self.stages))

    @property
    def causality(self) -> Causality:
        """causality Causality of the system

        Returns:
            Causality: Causality of the system
        """
        if self.causal:
            return Causality.CAUSAL
        return Causality.ANTICAUSAL
    
    @property
    def dims_in(self) -> List[int]:
        """dims_in Input dimensions for each time step

        Returns:
            List[int]: Input dimensions for each time step
        """
        return [el.dim_in for el in self.stages]

    @property
    def dims_out(self) -> List[int]:
        """dims_out Output dimensions for each time step

        Returns:
            List[int]: Output dimensions for each time step
        """
        return [el.dim_out for el in self.stages]

    @property
    def dims_state(self) -> List[int]:
        """dims_state State dimensions for each time step

        Returns:
            List[int]: State dimensions for each time step
        """
        return [el.dim_state for el in self.stages]

    def compute(self, input:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """compute Compute output of system for given input vector

        Args:
            input (np.ndarray): Input vector

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing the state vector and output vector
        """
        if self.causal:
            return self._compute_causal(input)
        return self._compute_anticausal(input)
    
    def to_matrix(self) -> np.ndarray:
        """to_matrix Create a matrix representation of the strict system.

        Returns:
            np.ndarray: Matrix representation
        """
        block_A = block_diag(*[el.A_matrix for el in self.stages])
        block_B = block_diag(*[el.B_matrix for el in self.stages])
        block_C = block_diag(*[el.C_matrix for el in self.stages])
        block_D = block_diag(*[el.D_matrix for el in self.stages])

        projection_1 = np.vstack([
            np.eye(block_A.shape[0]),
            np.zeros((1,block_A.shape[0]))
        ])
        projection_2 = np.hstack([
            np.zeros((block_A.shape[1],1)),
            np.eye(block_A.shape[1])
        ])

        A = projection_1@block_A@projection_2
        B = projection_1@block_B
        C = block_C@projection_2
        D = block_D
        Z = np.diag(*np.ones((1,A.shape[0]-1)),-1)

        return D + C@np.linalg.pinv(np.eye(A.shape[0]) - Z@A)@Z@B
    
    def _compute_causal(self, input:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """_compute_causal Compute output of causal system for given input vector

        Args:
            input (np.ndarray): Input vector

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing the state vector and output vector
        """
        k = len(self.stages)
        x_vectors = [np.zeros((0,1))]
        y_vectors = []
        in_index = 0
        for i in range(k):
            stage = self.stages[i]
            in_index_next = in_index + stage.dim_in
            u_in = input[in_index:in_index_next]
            in_index = in_index_next
            x_vectors.append(stage.A_matrix@x_vectors[i] + stage.B_matrix@u_in)
            y_vectors.append(stage.C_matrix@x_vectors[i] + stage.D_matrix@u_in)
        return (
            np.vstack(x_vectors),
            np.vstack(y_vectors))
    
    def _compute_anticausal(self, input:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """_compute_anticausal Compute output of anticausal system for given input vector

        Args:
            input (np.ndarray): Input vector

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing the state vector and output vector
        """
        k = len(self.stages)
        x_vectors = [np.zeros((0,1))]*(k+1)
        y_vectors = []
        in_index = len(input)
        for i in range(k-1,-1,-1):
            stage = self.stages[i]
            in_index_next = in_index - stage.dim_in
            u_in = input[in_index_next:in_index]
            in_index = in_index_next
            x_vectors[i] = stage.A_matrix@x_vectors[i+1] + stage.B_matrix@u_in
            y_vectors.append(stage.C_matrix@x_vectors[i+1] + stage.D_matrix@u_in)
        y_vectors.reverse()
        return (
            np.vstack(x_vectors),
            np.vstack(y_vectors))
    
    @staticmethod
    def zero(causal:bool, dims_in:List[int], dims_out:List[int]):
        """zero Generate empty system with given input/output dimensions

        Args:
            causal (bool): If true the system is causal, otherwise its anticausal
            dims_in (List[int]): Size of input dimensions
            dims_out (List[int]): Size of output dimensions

        Returns:
            [type]: Empty strict system
        """
        k = len(dims_in)
        stages = []
        for i in range(k):
            stages.append(Stage(
                np.zeros((0,0)),
                np.zeros((0,dims_in[i])),
                np.zeros((dims_out[i],0)),
                np.zeros((dims_out[i],dims_in[i]))))
        return StrictSystem(causal=causal,stages=stages)