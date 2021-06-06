from tvsclib.mixed_system import MixedSystem
from tvsclib.causality import Causality
import numpy as np
from tvsclib.expression import Expression
from tvsclib.system_interface import SystemInterface
from tvsclib.expressions.strict.multiply import multiply as multiplyStrict
from tvsclib.expressions.mixed.multiply import multiply as multiplyMixed
from tvsclib.expressions.utils.convert import convert
from tvsclib.expressions.transpose import Transpose
from tvsclib.expressions.invert import Invert

class Multiply(Expression):
    def __init__(self, lhs:Expression, rhs:Expression, name:str = "multiplication"):
        """__init__ Constructor

        Args:
            lhs (Expression): Left hand side expression
            rhs (Expression): Right hand side expression
            name (str, optional): Name of the expression. Defaults to "multiplication".
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
        return self.lhs.compute(self.rhs.compute(input))
    
    def transpose(self) -> Expression:
        """transpose Can be overwritten by concrete expression classes to
        carry out the transposition higher up in the expression tree if possible.

        Returns:
            Expression: An equivalent expression with the transposition moved to the operand(s)
            if possible, None otherwise
        """
        return Multiply(Transpose(self.rhs), Transpose(self.lhs))
    
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
        return Multiply(Invert(self.rhs), Invert(self.lhs))
    
    def realize(self) -> SystemInterface:
        """realize Generates a state space system from the expression tree

        Returns:
            SystemInterface: State space system
        """
        system_lhs = self.lhs.realize()
        system_rhs = self.rhs.realize()

        if system_lhs.causality is system_rhs.causality \
            and system_rhs.causality is not Causality.MIXED:
            return multiplyStrict(system_lhs, system_rhs)
        else:
            return multiplyMixed(
                convert(system_lhs, MixedSystem),
                convert(system_rhs, MixedSystem))
        
    
    def compile(self) -> Expression:
        """compile Returns an efficiently computeable expression tree

        Returns:
            Expression: Expression tree which may needs less memory and time
            to compute
        """
        return Multiply(self.lhs.compile(), self.rhs.compile())