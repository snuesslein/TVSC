import numpy as np
from typing import Callable
from tvsclib.mixed_system import MixedSystem
from tvsclib.expression import Expression
from tvsclib.system_interface import SystemInterface
from tvsclib.expressions.strict.invert import invert as invertStrict
from tvsclib.expressions.mixed.invert import invert as invertMixed
from tvsclib.expressions.const import Const

class Invert(Expression):
    def __init__(self, operand:Expression, name:str = "÷", compile_to_const:bool = False):
        """__init__ Constructor

        Args:
            operand (Expression): Operand expression
            name (str, optional): Name of the expression. Defaults to "inversion".
            compile_to_const (bool, optional): Set to True if compile shall generate a Const Expression.
                Otherwise the inversion gets compiled into a Multiply Expression. Defaults to False.
        """
        super().__init__(name, [operand])
        self.operand = operand
        self.compile_to_const = compile_to_const
    
    def compute(self, input:np.ndarray) -> np.ndarray:
        """compute Compute output of expression for given input vector.

        Args:
            input (np.ndarray): Input vector

        Returns:
            np.ndarray: Output vector
        """
        system = self.operand.realize()
        if type(system) is MixedSystem:
            return invertMixed(system).compute(input)
        else:
            return invertStrict(system).compute(input)
    
    def transpose(self, make_transpose:Callable[[Expression], Expression]) -> Expression:
        """transpose Can be overwritten by concrete expression classes to
        carry out the transposition lower down in the expression tree if possible.

        Args:
            make_transpose (Callable[[Expression], Expression]): Function that returns the transposed expression of the argument

        Returns:
            Expression: An equivalent expression with the transposition moved to the operand(s)
            if possible, None otherwise
        """
        return Invert(make_transpose(self.operand))
    
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
        return self.operand
    
    def realize(self) -> SystemInterface:
        """realize Generates a state space system from the expression tree

        Returns:
            SystemInterface: State space system
        """
        system = self.operand.realize()
        if type(system) is MixedSystem:
            return invertMixed(system).realize()
        else:
            return invertStrict(system).realize()
        
    def simplify(self) -> Expression:
        """simplify Returns a simplified expression tree

        Returns:
            Expression: Simplified expression tree
        """
        expr = self.operand.simplify()
        inv = expr.invert(lambda operand: Invert(operand))
        if inv is not None:
            return inv.simplify()
        return Invert(expr)
    
    def compile(self) -> Expression:
        """compile Returns a directly computeable expression tree

        Returns:
            Expression: Expression tree which may needs less memory and time
            to compute
        """
        if self.compile_to_const:
            inv = Invert(self.operand.compile())
            return Const(inv.realize(), f"({self.operand.name})^-1")
        system = self.operand.realize()
        if type(system) is MixedSystem:
            return invertMixed(system)
        else:
            return invertStrict(system)

