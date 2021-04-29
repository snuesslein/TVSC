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

R_causal = tvsc.RealizationStrict(causal=True,transferoperator=T,separation=sep)
T_rec_causal = R_causal.generate_transferoperator()

R_anticausal = tvsc.RealizationStrict(causal=False,transferoperator=T,separation=sep)
T_rec_anticausal = R_anticausal.generate_transferoperator()

u = np.array([1,2,3,4,5,6]).reshape((6,1))
(x_causal,y_causal) = R_causal.compute(u)
(x_anticausal,y_anticausal) = R_anticausal.compute(u)
