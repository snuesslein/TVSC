""" Definition of the realization strict (causal or anticausal) class. """
import numpy as np
from scipy.linalg import block_diag
from tvsclib import TransferOperator
from tvsclib.interfaces.statespace_interface import Causality
from tvsclib.interfaces.separation_interface import SeparationInterface
from tvsclib.interfaces.realization_interface import RealizationInterface

class RealizationStrict(RealizationInterface):
    """ Represents a strict causal or anticausal realization in state space.

    Attributes:
        causality (enum): Type of the system, either causal or anticausal
        dims_in (int[]): Input dimensions for each time step
        dims_out (int[]): Output dimensions for each time step
        dyn_degree (int[]): Dimension of the state space for each time step
        A (matrix[]): A matricies for each time step
        B (matrix[]): B matricies for each time step
        C (matrix[]): C matricies for each time step
        D (matrix[]): D matricies for each time step
    
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
        return [el.shape[0] for el in self.A]

    def __init__(self,causal=True,A=[],B=[],C=[],D=[],transferoperator=None,separation=None):
        """ Constructor.

        Args:
            causal (bool): Determines if the realization is causal or anticausal
            A (matrix[]): List of A matricies
            B (matrix[]): List of B matricies
            C (matrix[]): List of C matricies
            D (matrix[]): List of D matricies
            transferoperator (TransferOperator): Transfer operator instance which shall be used to generate realization
            separation (SeparationInterface): Separation object which shall be used to decompose transfer operator
        """
        self.causal = causal
        if transferoperator is not None and separation is not None:
            assert issubclass(type(transferoperator), TransferOperator)
            assert issubclass(type(separation), SeparationInterface)
            self.A,self.B,self.C,self.D = separation.separate(transferoperator,causal)
        else:
            self.A = A
            self.B = B
            self.C = C
            self.D = D

    def compute(self,u):
        """ Computes the result of a vector applied to this realization.

        Args:
            u (float[]): Vector which is applied

        Returns:
            x,y: Resulting state vector x and result vector y
        """
        if self.causal:
            return self._compute_causal(u)
        return self._compute_anticausal(u)

    def compile(self):
        return self

    def realize(self):
        return self

    def generate_transferoperator(self):
        """ Generates a transfer operator from the state space realization.

        Returns:
            TransferOperator: Transfer operator object
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