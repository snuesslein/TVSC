""" Definition of the realization strict (causal or anticausal) class. """
import numpy as np
from scipy.linalg import block_diag
from tvsclib.transfer_operator import TransferOperator
from tvsclib.causality import Causality
from tvsclib.interfaces.separation_interface import SeparationInterface
from tvsclib.interfaces.realization_interface import RealizationInterface

class RealizationStrict(RealizationInterface):
    """ Represents a strict causal or anticausal realization in state space.

    Attributes:
        A: A matricies for each time step.
        B: B matricies for each time step.
        C: C matricies for each time step
        D: D matricies for each time step.
    """

    @property
    def causality(self):
        if self.causal:
            return Causality.CAUSAL
        return Causality.ANTICAUSAL

    @property
    def dims_in(self):
        return [el.shape[1] for el in self.B]

    @property
    def dims_out(self):
        return [el.shape[0] for el in self.C]

    @property
    def dyn_degree(self):
        """ Dynamical degree of the system. """
        return [el.shape[0] for el in self.A]

    def __init__(
        self,
        causal:bool=True,
        A:list=None,B:list=None,C:list=None,D:list=None,
        transferoperator:TransferOperator=None,
        separation:SeparationInterface=None):
        """ Constructor.

        Args:
            causal: Determines if the realization is causal or anticausal.
            A: List of A matricies.
            B: List of B matricies.
            C: List of C matricies.
            D: List of D matricies.
            transferoperator: Transfer operator instance
                              which shall be used to generate realization.
            separation: Separation object which shall
                        be used to decompose transfer operator.
        """
        super().__init__()
        self.causal = causal
        if transferoperator is not None and separation is not None:
            self.A,self.B,self.C,self.D = separation.separate(transferoperator,causal)
        else:
            self.A = A
            self.B = B
            self.C = C
            self.D = D

    def _compute_function(self,u):
        """ Computes the result of a vector applied to this realization.

        Args:
            u: Vector which is applied.

        Returns:
            Resulting state vector x and result vector y.
        """
        if self.causal:
            return self._compute_causal(u)
        return self._compute_anticausal(u)

    def realize(self):
        return self

    def transform(self,transformation:str,**kwargs):
        """ Apply state transformation.

        Args:
            transformation: Name of the transformation.
            **kwargs: Arguments for the specific transformation.

        Returns:
            A realization with the same input output behaviour but different
            state space.
        """
        raise NotImplementedError("Not implemented yet")

    def generate_transferoperator(self):
        """ Generates a transfer operator from the state space realization.

        Returns:
            Transfer operator object.
        """
        A_blk = block_diag(*self.A)
        B_blk = block_diag(*self.B)
        C_blk = block_diag(*self.C)
        D_blk = block_diag(*self.D)

        projection_1 = np.vstack([
            np.eye(A_blk.shape[0]),
            np.zeros((1,A_blk.shape[0]))
        ])
        projection_2 = np.hstack([
            np.zeros((A_blk.shape[1],1)),
            np.eye(A_blk.shape[1])
        ])

        A = projection_1@A_blk@projection_2
        B = projection_1@B_blk
        C = C_blk@projection_2
        D = D_blk
        Z = RealizationInterface.shiftoperator(A.shape[0])

        matrix = D + C@np.linalg.pinv(np.eye(A.shape[0]) - Z@A)@Z@B

        return TransferOperator(matrix, self.dims_in, self.dims_out)

    def _compute_causal(self,u):
        k = len(self.A)
        x_vectors = [np.zeros((0,1))]
        y_vectors = []
        in_index = 0
        for i in range(k):
            in_index_next = in_index + self.dims_in[i]
            u_in = u[in_index:in_index_next]
            in_index = in_index_next
            x_vectors.append(self.A[i]@x_vectors[i] + self.B[i]@u_in)
            y_vectors.append(self.C[i]@x_vectors[i] + self.D[i]@u_in)
        return (
            np.vstack(x_vectors),
            np.vstack(y_vectors)
        )

    def _compute_anticausal(self,u):
        k = len(self.A)
        x_vectors = [np.zeros((0,1))]*(k+1)
        y_vectors = []
        in_index = len(u)
        for i in range(k,0,-1):
            in_index_next = in_index - self.dims_in[i-1]
            u_in = u[in_index_next:in_index]
            in_index = in_index_next
            x_vectors[i-1] = self.A[i-1]@x_vectors[i] + self.B[i-1]@u_in
            y_vectors.append(self.C[i-1]@x_vectors[i] + self.D[i-1]@u_in)
        y_vectors.reverse()
        return (
            np.vstack(x_vectors),
            np.vstack(y_vectors)
        )

    @staticmethod
    def zero(causal:bool,dims_in,dims_out):
        """ Generates a zero realization.

        Args:
            causal: Causality of the system.
            dims_in: Input dimensions.
            dims_out: Output dimensions.

        Returns:
            Zero realization.
        """
        result_A = []
        result_B = []
        result_C = []
        result_D = []
        k = len(dims_in)
        for i in range(k):
            result_A.append(np.zeros((0,0)))
            result_B.append(np.zeros((0,dims_in[i])))
            result_C.append(np.zeros((dims_out[i],0)))
            result_D.append(np.zeros((dims_out[i],dims_in[i])))
        return RealizationStrict(
            causal=causal,
            A = result_A,
            B = result_B,
            C = result_C,
            D = result_D
        )
