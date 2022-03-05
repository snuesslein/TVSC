import numpy as np

def hankelnorm(A, dims_in,dims_out):
    """calculates the Hankel Norm

    calculates the Hankel Norm of the Matrix A with the given segementation.
    The Hankel Norm is the largest singular value of the Hankel operators

    The formal defineition is:
    ||T||_H = sup_{i}||T_i||
    Where ||T_i|| is rthe spectral norm of the Hankel operators

        Args:
            A (numpy.ndarray):      Matrix to calculate the hankel norm
            dims_in (List[int]):    input dimension
            dims_out (List[int]):   output dimension

        Returns:
            float:  Hankel norm of A
    """

    #for more details on the implementation see the notebook on Obs and Reach
    n = len(dims_in)
    s_c = [np.max(np.linalg.svd(A[-np.sum(dims_out[k:]):,:np.sum(dims_in[:k])],compute_uv=False)) for k in range(1,n)]
    s_a = [np.max(np.linalg.svd(A[:np.sum(dims_out[:k+1]),-np.sum(dims_in[k+1:]):],compute_uv=False)) for k in range(n-2,-1,-1)]
    return max(max(s_c),max(s_a))


def cost(dims_in,dims_out,dims_state,causal,include_add=False):
    """calculates the computational cost

    This return the FLOPs needed to calcualte the output for a input vector

    p = dims_in,
    m = dims_out,
    n = dims_state,

    Without additions:

    n_{k+1}*n_k + n{k+1}*m_k+p_k*n_k+p_k*m_k

    With additions:

    n_{k+1}*(2*n_k-1) + n{k+1}*(2*m_k-1)+p_k*(2*n_k-1)+p_k*(2*m_k-1)

    The case without additions is equal to the number of parameters

        Args:
            dims_in (List[int]):    input dimension
            dims_out (List[int]):   output dimension
            dims_state (List[int]): state dimension

        Returns:
            int:  Number of FLOPs
    """

    if causal:
        m = np.array(dims_in)
        p = np.array(dims_out)
        n = np.array(dims_state)
    else: #reverse them for the anticausal part, then we can use the same formula
        m = np.array(dims_in[::-1])
        p = np.array(dims_out[::-1])
        n = np.array(dims_state[::-1])
    if include_add:
        return np.sum(n[1:]*(2*n[:-1]-1) + n[1:]*(2*m-1)+p*(2*n[:-1]-1)+p*(2*m-1))
    else:
        return np.sum(n[1:]*n[:-1] + n[1:]*m+p*n[:-1]+p*m)
