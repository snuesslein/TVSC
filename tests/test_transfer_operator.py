from tvsclib import TransferOperator
import numpy as np

def test_get_hankels():
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
    hankels_causal_expected = [
        np.zeros((0,0)),
        np.array([
            [2, 3],
            [6, 3],
            [3, 5],
            [2, 4],
            [2, 4],
        ]),
        np.array([
            [5, 3, 5],
            [3, 2, 4],
            [4, 2, 4]
        ]),
        np.array([
            [6, 1, 3, 2, 4],
            [1, 5, 4, 2, 4]
        ])
    ]
    hankels_anticausal_expected = [
        np.array([
            [6,     1,     4,     2]
        ]),
        np.array([
            [1, 3, 4],
            [4, 1, 1],
            [1, 4, 2]
        ]),
        np.array([
            [4],
            [4],
            [1],
            [2]
        ]),
        np.zeros((0,0))
    ]
    T = TransferOperator(matrix, dims_in, dims_out)
    hankels_causal = T.get_hankels(True)
    hankels_anticausal = T.get_hankels(False)
    assert all([np.all(i == j) for i, j in zip(hankels_causal,hankels_causal_expected)])
    assert all([np.all(i == j) for i, j in zip(hankels_anticausal,hankels_anticausal_expected)])