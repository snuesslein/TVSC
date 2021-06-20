from tvsclib.strict_system import StrictSystem
from tvsclib.expressions.const import Const
from tvsclib.expressions.multiply import Multiply

def invert(system:StrictSystem) -> Multiply:
    """invert Inversion in state space

    Args:
        system (StrictSystem): System to invert

    Returns:
        Multiply: Expression which computes inverese
    """
    if system.causal:
        T_ol, V_r = system.outer_inner_factorization()
        V_l, T_o = T_ol.inner_outer_factorization()
        T_o_inverse = T_o.arrow_reversal()
        result = Multiply(
            Const(V_r.transpose(), "V_r'"),
            Const(T_o_inverse, "T_o^-1"),
            "V_r'*T_o^-1")
        result = Multiply(
            result,
            Const(V_l.transpose(), result.name+"*V_l'"))
        return result
    T_ol, V_r = system.transpose().outer_inner_factorization()
    V_l, T_o = T_ol.inner_outer_factorization()
    T_o_inverse = T_o.arrow_reversal()
    result = Multiply(
        Const(V_l, "V_l"),
        Const(T_o_inverse.transpose(), "T_o'^-1"),
        "V_l*T_o'^-1")
    result = Multiply(
        result,
        Const(V_r, "V_r"),
        result.name+"*V_r")
    return result