""" Class definition of TransferOperator. """
import numpy as np


class TransferOperator:
    """ Represents a simple linear transfer operator with inputs and outputs.

    Attributes:
        matrix: Transfer operator matrix.
        dims_in: input dimensions at each timestep as a list of integers.
        dims_out: output dimensions at each timestep as list of integers.

    """
    def __init__(self, matrix, dims_in, dims_out):
        """ Constructor.

        Args:
            matrix: Transfer operator matrix.
            dims_in: input dimensions at each timestep as a list of integers.
            dims_out: output dimensions at each timestep as list of integers.
        """
        self.matrix   = matrix
        self.dims_in  = dims_in
        self.dims_out = dims_out

    def get_hankels(self, causal: bool):
        """ Extracting hankel matricies from transfer operator.

        Get the causal or anticausal hankel matricies that define
        the input/output mapping of the transfer operator.

        Args:
            causal: specifies if the causal or anticausal matricies shall be returned.

        Returns:
            A list of matricies.
        """
        if causal:
            number_of_inputs = len(self.dims_in)
            hankels = [np.zeros((0,0))]
            for i in range(1,number_of_inputs):
                blocks = []
                for k in range(0,i):
                    rows = range(sum(self.dims_out[0:i]), sum(self.dims_out))
                    cols = range(sum(self.dims_in[0:k]), sum(self.dims_in[0:k+1]))
                    blocks.append(self.matrix[np.ix_(rows,cols)])
                blocks.reverse()
                hankels.append(np.hstack(blocks))
        else:
            number_of_outputs = len(self.dims_out)
            hankels = []
            for i in range(0,number_of_outputs-1):
                blocks = []
                for k in range(0,i+1):
                    rows = range(sum(self.dims_out[0:k]), sum(self.dims_out[0:k+1]))
                    cols = range(sum(self.dims_in[0:i+1]), sum(self.dims_in))
                    blocks.append(self.matrix[np.ix_(rows,cols)])
                blocks.reverse()
                hankels.append(np.vstack(blocks))
            hankels.append(np.zeros((0,0)))
        return hankels
