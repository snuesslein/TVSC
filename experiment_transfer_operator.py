import tvsclib as tvsc
import numpy as np

matrix = np.random.rand(6,6)
dims_in = [1,1,1,1,1,1]
dims_out = [1,1,1,1,1,1]
T = tvsc.TransferOperator(matrix,dims_in,dims_out)
hankels = T.get_hankels(True)
for h in hankels:
    print(h)
