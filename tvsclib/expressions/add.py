import numpy as np
from tvsclib.mixed_system import MixedSystem
from tvsclib.expression import Expression
from tvsclib.system_interface import SystemInterface
from tvsclib.expressions.strict.add import add as addStrict
from tvsclib.expressions.mixed.add import add as addMixed
from tvsclib.expressions.utils.convert import convert
from tvsclib.expressions.transpose import Transpose

class Add(Expression):
    def __init__(self, lhs:Expression, rhs:Expression, name:str = "addition"):
        """__init__ Constructor

        Args:
            lhs (Expression): Left hand side expression
            rhs (Expression): Right hand side expression
            name (str, optional): Name of the expression. Defaults to "addition".
        """
        super().__init__(name, [lhs, rhs])
        self.lhs = lhs
        self.rhs = rhs
    
    def compute(self, input:np.ndarray) -> np.ndarray:
        """compute Compute output of expression for given input vector.

        Args:
            input (np.ndarray): Input vector

        Returns:
            np.ndarray: Output vector
        """
        return self.lhs.compute(input) + self.rhs.compute(input)
    
    def transpose(self) -> Expression:
        """transpose Can be overwritten by concrete expression classes to
        carry out the transposition higher up in the expression tree if possible.

        Returns:
            Expression: An equivalent expression with the transposition moved to the operand(s)
            if possible, None otherwise
        """
        return Add(Transpose(self.lhs), Transpose(self.rhs))
    
    def realize(self) -> SystemInterface:
        """realize Generates a state space system from the expression tree

        Returns:
            SystemInterface: State space system
        """
        system_lhs = self.lhs.realize()
        system_rhs = self.rhs.realize()

        if type(system_lhs) is type(system_rhs) \
            and type(system_rhs) is not MixedSystem:
            return addStrict(system_lhs, system_rhs)
        else:
            return addMixed(
                convert(system_lhs, MixedSystem),
                convert(system_rhs, MixedSystem))
        
    
    def compile(self) -> Expression:
        """compile Returns an efficiently computeable expression tree

        Returns:
            Expression: Expression tree which may needs less memory and time
            to compute
        """
        return Add(self.lhs.compile(), self.rhs.compile())