import numpy as np
from typing import Callable
from tvsclib.mixed_system import MixedSystem
from tvsclib.expression import Expression
from tvsclib.system_interface import SystemInterface
from tvsclib.expressions.strict.transpose import transpose as transposeStrict
from tvsclib.expressions.mixed.transpose import transpose as transposeMixed
from tvsclib.expressions.const import Const

class Transpose(Expression):
    def __init__(self, operand:Expression, name:str = "transposition"):
        """__init__ Constructor

        Args:
            operand (Expression): Operand expression
            name (str, optional): Name of the expression. Defaults to "transposition".
        """
        super().__init__(name, [operand])
        self.operand = operand
    
    def compute(self, input:np.ndarray) -> np.ndarray:
        """compute Compute output of expression for given input vector.

        Args:
            input (np.ndarray): Input vector

        Returns:
            np.ndarray: Output vector
        """
        _, y = self.realize().compute(input)
        return y
    
    def transpose(self, make_transpose:Callable[[Expression], Expression]) -> Expression:
        """transpose Can be overwritten by concrete expression classes to
        carry out the transposition lower down in the expression tree if possible.

        Args:
            make_transpose (Callable[[Expression], Expression]): Function that returns the transposed expression of the argument

        Returns:
            Expression: An equivalent expression with the transposition moved to the operand(s)
            if possible, None otherwise
        """
        return self.operand
    
    def invert(self, make_inverse:Callable[[Expression], Expression]) -> Expression:
        """invert Can be overwritten by concrete expression classes to
        carry out the inversion lower down in the expression tree if possible.

        E.g. ((A + B) * C)^1 -> C^-1 * (A + B)^-1. Since we are usually loosing minimality
        when doing additions or multiplications the state space gets rather large.
        Computing the inverse on this "bloated" state space is computational costly. Therefor
        it is better to carry out the inversion earlier on "more minimal" systems.

        Args:
            make_inverse (Callable[[Expression], Expression]): Function that returns the inverse expression of the argument

        Returns:
            Expression: An equivalent expression with the inversion moved to the operand(s)
            if possible, None otherwise
        """
        return None
    
    def realize(self) -> SystemInterface:
        """realize Generates a state space system from the expression tree

        Returns:
            SystemInterface: State space system
        """
        system = self.operand.realize()
        if type(system) is MixedSystem:
            return transposeMixed(system)
        else:
            return transposeStrict(system)
        
    def simplify(self) -> Expression:
        """simplify Returns a simplified expression tree

        Returns:
            Expression: Simplified expression tree
        """
        expr = self.operand.simplify()
        trp = expr.transpose(lambda operand: Transpose(operand).simplify())
        if trp is not None:
            return trp.simplify()
        return Transpose(expr.simplify())

    def compile(self) -> Expression:
        """compile Returns a directly computeable expression tree

        Returns:
            Expression: Expression tree which may needs less memory and time
            to compute
        """
        trp = Transpose(self.operand.compile())
        return Const(trp.realize(), f"({self.operand.name})'")
