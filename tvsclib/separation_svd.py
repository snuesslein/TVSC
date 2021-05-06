""" Definition of the singular value decomposition separation class. """
from enum import Enum
import numpy as np
from tvsclib.interfaces.separation_interface import SeparationInterface


class NormalForm(Enum):
    """ Normal Forms of a state space system. """
    INPUT = 1
    OUTPUT = 2
    BALANCED = 3

class SeparationSVD(SeparationInterface):
    """ Can be used to find Observability and Controlability matricies via SVD.

    Attributes:
        epsilon: Lower limit for singular values.
        relative: Indication if epsilon value is absolute or realtive to sum of all
                  singular values.
        form: Specification which output-normal form to use.

    """

    def __init__(self, epsilon: float, relative=True, form=NormalForm.BALANCED):
        """ Constructor.

        Args:
            epsilon: Lower limit for singular values.
            relative: Indication if epsilon value is absolute or
                      realtive to sum of all singular values.
            form: Specification which output-normal form to use.
        """
        self.epsilon = epsilon
        self.relative = relative
        self.form = form

    def factorize_hankel(self, hankel):
        """ Factorizes a hankel matrix into observability and controlability matrix.

        Args:
            hankel: Hankel matrix

        Returns:
            Observability and controlability matrix
        """
        number_of_rows,number_of_cols = hankel.shape
        if number_of_rows >= number_of_cols:
            U,S,VH = np.linalg.svd(hankel)
            V = VH.transpose()
        else:
            V,S,UH = np.linalg.svd(hankel.transpose())
            U = UH.transpose()
        # Rank approximation
        rank = len(S)
        singular_values_total = sum(S)
        if self.relative:
            rank_approx = 0
            while rank_approx < rank:
                if sum(S[0:rank_approx]) / singular_values_total >= 1 - self.epsilon:
                    break
                rank_approx = rank_approx + 1
        else:
            rank_approx = sum(S > self.epsilon)
        # Retrieving observability and controlability matrix
        (Obs,Con) = {
            NormalForm.OUTPUT: lambda U,S,V,rank_approx: (
                U[:,0:rank_approx],
                np.diag(S[0:rank_approx]) @ (V[:,0:rank_approx].transpose())
            ),
            NormalForm.INPUT: lambda U,S,V,rank_approx: (
                U[:,0:rank_approx] @ np.diag(S[0:rank_approx]),
                V[:,0:rank_approx].transpose()
            ),
            NormalForm.BALANCED: lambda U,S,V,rank_approx: (
                U[:,0:rank_approx] @ np.diag(np.sqrt(S[0:rank_approx])),
                np.diag(np.sqrt(S[0:rank_approx])) @ (V[:,0:rank_approx].transpose())
            )
        }[self.form](U,S,V,rank_approx)
        return (Obs,Con)
