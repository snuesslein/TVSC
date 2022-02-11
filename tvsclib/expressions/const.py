import numpy as np
from tvsclib.system_interface import SystemInterface
from tvsclib.expression import Expression

class Const(Expression):
    def __init__(self, operand:SystemInterface, name:str = "const"):
        """__init__ Constructor

        Args:
            operand (SystemInterface): System interface object
            name (str, optional): Name of this expression. Defaults to "constant".
        """
        super().__init__(name, [])
        self.operand = operand

    def compute(self, input:np.ndarray) -> np.ndarray:
        """compute Compute output of expression for given input vector.

        Args:
            input (np.ndarray): Input vector

        Returns:
            np.ndarray: Output vector
        """
        _, y = self.operand.compute(input)
        return y

    def realize(self) -> SystemInterface:
        """realize Generates a state space system from the expression tree

        Returns:
            SystemInterface: State space system
        """
        return self.operand
    
    def simplify(self) -> Expression:
        """simplify Returns a simplified expression tree

        Returns:
            Expression: Simplified expression tree
        """
        return self
    
    def compile(self) -> Expression:
        """compile Returns a directly computeable expression tree

        Returns:
            Expression: Expression tree which may needs less memory and time
            to compute
        """
        return self