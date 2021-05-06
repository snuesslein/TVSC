""" Definition of the separation interface. """
import abc
import numpy as np
from tvsclib.transfer_operator import TransferOperator

class SeparationInterface(object):
    __metaclass__ = abc.ABCMeta
    """ Classes which inherit this interface can be used to
    create state space matricies from a transfer operator.
    """
    def __init__(self):
        """ Constructor. """  
        pass

    def separate(self, transfer_operator:TransferOperator, causal:bool):
        """ Separates transfer operator into A,B,C and D.
        matricies.

        Args:
            transfer_operator: Transfer Operator for which A, B, C and D shall be found.
            causal: Determines if the causal or anticausal part is returned.

        Returns:
            Lists of A,B,C, and D matricies.
        """
        [Obs,Con] = self.factorize_hankels(transfer_operator, causal)
        if causal:
            ranks = [el.shape[0] for el in Con]
            number_of_outputs = len(transfer_operator.dims_out)
            A,B,C,D = ([],[],[],[])
            for k in range(0,number_of_outputs):
                rows = range(sum(transfer_operator.dims_out[0:k]), sum(transfer_operator.dims_out[0:k+1]))
                cols = range(sum(transfer_operator.dims_in[0:k]), sum(transfer_operator.dims_in[0:k+1]))
                D.append(transfer_operator.matrix[np.ix_(rows,cols)])
            A.append(np.zeros((ranks[1],0)))
            C.append(np.zeros((transfer_operator.dims_out[0],0)))
            for k in range(1,number_of_outputs):
                C.append(Obs[k][0:transfer_operator.dims_out[k],:])
            for k in range(0,number_of_outputs-1):
                B.append(Con[k+1][:,0:transfer_operator.dims_in[k]])
            for k in range(1,number_of_outputs-1):
                ObsUp = Obs[k][transfer_operator.dims_out[k]:,:]
                A.append(np.linalg.pinv(Obs[k+1]) @ ObsUp)
            A.append(np.zeros((0,ranks[number_of_outputs-1])))
            B.append(np.zeros((0,transfer_operator.dims_in[number_of_outputs-1])))
        else:
            ranks = [el.shape[1] for el in Obs]
            number_of_inputs = len(transfer_operator.dims_in)
            A,B,C,D = ([],[],[],[])
            for k in range(0,number_of_inputs):
                D.append(np.zeros((transfer_operator.dims_out[k],transfer_operator.dims_in[k])))
            A.append(np.zeros((0,ranks[0])))
            B.append(np.zeros((0,transfer_operator.dims_in[0])))
            for k in range(1,number_of_inputs):
                B.append(Con[k-1][:,0:transfer_operator.dims_in[k]])
            for k in range(0,number_of_inputs-1):
                C.append(Obs[k][0:transfer_operator.dims_out[k],:])
            for k in range(1,number_of_inputs-1):
                ObsUp = Obs[k][transfer_operator.dims_out[k]:,:]
                A.append(np.linalg.pinv(Obs[k-1]) @ ObsUp)
            A.append(np.zeros((ranks[number_of_inputs-2],0)))
            C.append(np.zeros((transfer_operator.dims_out[number_of_inputs-1],0)))
        return (A,B,C,D)
    
    def factorize_hankels(self, transfer_operator:TransferOperator, causal:bool):
        """ Factorizes hankel matricies into observability and controlability matricies.

        Args:
            transfer_operator: Transfer Operator for which observability and controlability matricies shall be found.
            causal: Determines if the causal or anticausal part is returned.

        Returns:
            Lists of reachability and controlability matricies.
        """
        number_of_inputs = len(transfer_operator.dims_in)
        number_of_outputs = len(transfer_operator.dims_out)
        hankels = transfer_operator.get_hankels(causal)
        Obs = []
        Con = []
        if causal:
            Obs.append(np.zeros((0,0)))
            Con.append(np.zeros((0,0)))
            for i in range(1,number_of_outputs):
                [Obs_i,Con_i] = self.factorize_hankel(hankels[i])
                Obs.append(Obs_i)
                Con.append(Con_i)
        else:
            for i in range(0,number_of_inputs-1):
                [Obs_i,Con_i] = self.factorize_hankel(hankels[i])
                Obs.append(Obs_i)
                Con.append(Con_i)
            Obs.append(np.zeros((0,0)))
            Con.append(np.zeros((0,0)))
        return (Obs,Con)

    @abc.abstractmethod
    def factorize_hankel(self, hankel):
        """ Factorizes a hankel matrix into observability and controlability matrix.

        Args:
            hankel: Hankel matrix.
        
        Returns:
            Observability and controlability matrix.
        """
        pass