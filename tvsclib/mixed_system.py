from __future__ import annotations
import numpy as np
from typing import Tuple, List
from tvsclib.causality import Causality
from tvsclib.strict_system import StrictSystem
from tvsclib.system_identification_interface import SystemIdentificationInterface
from tvsclib.system_interface import SystemInterface

class MixedSystem(SystemInterface):
    def __init__(self,
        system_identification:SystemIdentificationInterface = None,
        causal_system:StrictSystem = None, anticausal_system:StrictSystem = None):
        if system_identification is not None:
            self.causal_system = StrictSystem(causal=True,system_identification=system_identification)
            self.anticausal_system = StrictSystem(causal=False,system_identification=system_identification)
        elif causal_system is not None and anticausal_system is not None:
            self.causal_system = causal_system
            self.anticausal_system = anticausal_system
        else:
            raise AttributeError("Not enough arguments provided")
    
    @property
    def causality(self) -> Causality:
        """causality Causality of the system

        Returns:
            Causality: Causality of the system
        """
        return Causality.MIXED
    
    @property
    def dims_in(self) -> List[int]:
        """dims_in Input dimensions for each time step

        Returns:
            List[int]: Input dimensions for each time step
        """
        return self.causal_system.dims_in

    @property
    def dims_out(self) -> List[int]:
        """dims_out Output dimensions for each time step

        Returns:
            List[int]: Output dimensions for each time step
        """
        return self.causal_system.dims_out
    
    def copy(self) -> MixedSystem:
        """copy Returns a copy of this system

        Returns:
            MixedSystem: Copy of this system
        """
        return MixedSystem(
            causal_system=self.causal_system.copy(),
            anticausal_system=self.anticausal_system.copy())
    
    def to_matrix(self) -> np.ndarray:
        """to_matrix Create a matrix representation of the mixed system.

        Returns:
            np.ndarray: Matrix representation
        """
        return self.causal_system.to_matrix() + self.anticausal_system.to_matrix()

    def compute(self, input:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """compute Compute output of system for given input vector.
        The states of the causal and anticausal system are returned in stacked
        fashion [x_causal,x_anticausal]'.

        Args:
            input (np.ndarray): Input vector

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing the state vector and output vector
        """
        x_causal,y_causal = self.causal_system.compute(input)
        x_anticausal,y_anticausal = self.anticausal_system.compute(input)
        x_result = np.vstack([
            x_causal, x_anticausal
        ])
        y_result = y_causal + y_anticausal
        return (x_result,y_result)