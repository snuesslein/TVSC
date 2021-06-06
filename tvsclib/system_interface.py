from __future__ import annotations
import numpy as np
from typing import List, Tuple
from tvsclib.causality import Causality

class SystemInterface:
    def copy(self) -> SystemInterface:
        """copy Returns a copy of this system

        Returns:
            SystemInterface: Copy of this system
        """
        raise NotImplementedError("copy not implemented")
    
    def compute(self, input:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """compute Compute output of system for given input vector.

        Args:
            input (np.ndarray): Input vector

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing the state vector and output vector
        """

    def to_matrix(self) -> np.ndarray:
        """to_matrix Create a matrix representation of the system.

        Returns:
            np.ndarray: Matrix representation
        """
        raise NotImplementedError("copy not implemented")
    
    @property
    def causality(self) -> Causality:
        """causality Causality of the system

        Returns:
            Causality: Causality of the system
        """
        raise NotImplementedError("causality not implemented")
    
    @property
    def dims_in(self) -> List[int]:
        """dims_in Input dimensions for each time step

        Returns:
            List[int]: Input dimensions for each time step
        """
        raise NotImplementedError("dims_in not implemented")

    @property
    def dims_out(self) -> List[int]:
        """dims_out Output dimensions for each time step

        Returns:
            List[int]: Output dimensions for each time step
        """
        raise NotImplementedError("dims_out not implemented")
