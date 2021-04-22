import tvsclib as tvsc
import numpy as np


matrix = np.array([
    [5,     4,     6,     1,     4,     2],
    [2,     3,     2,     1,     3,     4],
    [6,     3,     5,     4,     1,     1],
    [3,     5,     5,     5,     3,     4],
    [2,     4,     3,     6,     1,     2],
    [2,     4,     4,     1,     5,     4]
])
dims_in =  [2, 1, 2, 1]
dims_out = [1, 2, 1, 2]

T = tvsc.TransferOperator(matrix,dims_in,dims_out)

sep = tvsc.SeparationSVD(0)

A,B,C,D = sep.separate(T,False)