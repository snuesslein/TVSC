from __future__ import annotations
import numpy as np
from typing import List
from tvsclib.system_interface import SystemInterface

class Expression:
    def __init__(self, name:str, childs:List[Expression]):
        """__init__ Constructor

        Args:
            name (str): Name of this expression
            childs (List[Expression]): Child expressions
        """
        self.name = name
        self.childs = childs
    

    def compute(self, input:np.ndarray) -> np.ndarray:
        """compute Compute output of expression for given input vector.

        Args:
            input (np.ndarray): Input vector

        Returns:
            np.ndarray: Output vector
        """
        raise NotImplementedError("compute not implemented")

    def realize(self) -> SystemInterface:
        """realize Generates a state space system from the expression tree

        Returns:
            SystemInterface: State space system
        """
        raise NotImplementedError("realize not implemented")
    
    def compile(self) -> Expression:
        """compile Returns an efficiently computeable expression tree

        Returns:
            Expression: Expression tree which may needs less memory and time
            to compute
        """
        raise NotImplementedError("compile not implemented")
    
    def invert(self) -> Expression:
        """invert Can be overwritten by concrete expression classes to
        carry out the inversion higher up in the expression tree if possible.

        E.g. ((A + B) * C)^1 -> C^-1 * (A + B)^-1. Since we are usually loosing minimality
        when doing additions or multiplications the state space gets rather large.
        Computing the inverse on this "bloated" state space is computational costly. Therefor
        it is better to carry out the inversion earlier on "more minimal" systems.

        Returns:
            Expression: An equivalent expression with the inversion moved to the operand(s)
            if possible, None otherwise
        """
        return None
    
    def transpose(self) -> Expression:
        """transpose Can be overwritten by concrete expression classes to
        carry out the transposition higher up in the expression tree if possible.

        Returns:
            Expression: An equivalent expression with the transposition moved to the operand(s)
            if possible, None otherwise
        """
        return None