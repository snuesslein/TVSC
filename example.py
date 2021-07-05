# %% Import neccessary stuff
import numpy as np
from graphviz import Digraph
from tvsclib.strict_system import StrictSystem
from tvsclib.mixed_system import MixedSystem
from tvsclib.toeplitz_operator import ToeplitzOperator
from tvsclib.system_identification_svd import SystemIdentificationSVD
from tvsclib.transformations.reduction import Reduction
from tvsclib.expressions.const import Const
from tvsclib.expressions.add import Add
from tvsclib.expressions.multiply import Multiply
from tvsclib.expressions.invert import Invert
from tvsclib.expressions.transpose import Transpose
from tvsclib.expression import Expression

# %% Helper function to generate expression tree graph
def build_graph(expression:Expression, f_out:Digraph):
    f_out.node(str(id(expression)), label=expression.name)
    op_counter = 1
    for child in expression.childs:
        build_graph(child, f_out)
        f_out.edge(str(id(expression)), str(id(child)), label=f"op:{op_counter}")
        op_counter = op_counter + 1

# %% Set up a matricies
dims_in =  [2, 1, 2, 1]
dims_out = [2, 1, 2, 1]
u = np.random.rand(np.sum(dims_in),1)
matrix_A = np.random.rand(np.sum(dims_out), np.sum(dims_in))
matrix_B = np.random.rand(np.sum(dims_out), np.sum(dims_in))
matrix_C = np.random.rand(np.sum(dims_out), np.sum(dims_in))

# %% Generate time varying systems via hankel factorization
T = ToeplitzOperator(matrix_A, dims_in, dims_out)
S = SystemIdentificationSVD(T)
system_A = MixedSystem(S)

T = ToeplitzOperator(matrix_B, dims_in, dims_out)
S = SystemIdentificationSVD(T)
system_B = StrictSystem(system_identification=S, causal=True)

T = ToeplitzOperator(matrix_C, dims_in, dims_out)
S = SystemIdentificationSVD(T)
system_C = MixedSystem(system_identification=S, causal_system=True)

# %% Compute states and output vector
x,y = system_A.compute(u)
print(f"Error norm: {np.linalg.norm(y - matrix_A @ u)}")

# %% Build an expression
A = Const(system_A, "A")
B = Const(system_B, "B")
C = Const(system_C, "C")

add = Add(A, B)
inv = Invert(add)
mul = Multiply(inv, C)
trp = Transpose(mul)

simplified_expression = trp.simplify()
compiled_expression = simplified_expression.compile()
realized_expression = simplified_expression.realize()
realized_expression_reduced = simplified_expression\
    .post_realize(lambda s: Reduction().apply(s), True)\
        .realize()

print(f"Is system minimal: {realized_expression_reduced.is_minimal()}")

# %% Generate graphs
my_graph = Digraph(directory="./")
build_graph(trp, my_graph)
my_graph.view(filename="expression.gv")

my_graph = Digraph(directory="./")
build_graph(simplified_expression, my_graph)
my_graph.view(filename="simplified_expression.gv")