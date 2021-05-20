""" Definition of the multiply mixed class. """
import numpy as np
from tvsclib.causality import Causality
from tvsclib.realization_strict import RealizationStrict
from tvsclib.interfaces.statespace_interface import StateSpaceInterface

class MultiplyMixed(StateSpaceInterface):
    """ Represents a multiplication operation in state space. """

    @property
    def causality(self):
        return Causality.MIXED

    def __init__(self,lhs_op:StateSpaceInterface,rhs_op:StateSpaceInterface):
        """ Constructor.

        Args:
            lhs_op: Left hand side operand.
            rhs_op: Right hand side operand.
        """
        super().__init__(self._compute_function)
        if lhs_op.causality is not rhs_op.causality:
            raise AttributeError("MultiplyMixed lhs_op and rhs_op have different causalities")
        if lhs_op.causality is not Causality.MIXED:
            raise AttributeError("MultiplyMixed can not handle strict systems")
        self.lhs_op = lhs_op
        self.rhs_op = rhs_op

    def transpose(self):
        """ Override transpose method.
        Computing rhs' * lhs' is equal to (lhs * rhs)'.

        returns:
            Transposed multiply operation.
        """
        return MultiplyMixed(self.rhs_op.transpose(),self.lhs_op.transpose())

    def _compute_function(self,u):
        """ Applies a vector to multiplication result in state space.

        Args:
            u: Vector to be applied.

        Returns: 
            Resulting state vector x and result vector y.
        """
        x_rhs,y_rhs = self.rhs_op.compute(u)
        x_lhs,y_lhs = self.lhs_op.compute(y_rhs)
        x_result = np.vstack([
            x_rhs,
            x_lhs
        ])
        y_result = y_lhs
        return (x_result,y_result)

    def realize(self):
        """ Generates a state space realization of the multiplication operation. 
        
        Returns:
            Realization of multiplication operation.
        """
        realization_lhs = self.lhs_op.realize()
        realization_rhs = self.rhs_op.realize()
        
        if ~np.all(realization_lhs.dims_in == realization_rhs.dims_out):
            raise AttributeError("Input/Output dimensions dont match")
        
        result = realization_lhs.causal_system.mul(
            realization_rhs.causal_system
        )
        result = result.add(
            realization_lhs.anticausal_system.mul(
                realization_rhs.anticausal_system
            )
        )
        result = result.add(
            self._causal_times_anticausal(
                realization_lhs.causal_system,
                realization_rhs.anticausal_system
            )
        )
        result = result.add(
            self._anticausal_times_causal(
                realization_lhs.anticausal_system,
                realization_rhs.causal_system
            )
        )
        return result.realize()

    def _causal_times_anticausal(
        self,
        causal_system:RealizationStrict,
        anticausal_system:RealizationStrict):
        M = [np.zeros((0,0))]
        k = len(causal_system.A)
        for i in range(k):
            M.append(
                causal_system.B[i] @ anticausal_system.C[i]
                + causal_system.A[i] @ M[i] @ anticausal_system.A[i]
            )
        # Compute strictly causal part of multiplication result
        causal_B = []
        causal_D = []
        for i in range(k):
            causal_B.append(
                causal_system.A[i] @ M[i] @ anticausal_system.B[i]
            )
            causal_D.append(
                causal_system.C[i] @ M[i] @ anticausal_system.B[i]
            )
        result = RealizationStrict(
            causal=True,
            A=causal_system.A,
            B=causal_B,
            C=causal_system.C,
            D=causal_D
        )
        # Compute strictly anticausal part of multiplication result
        anticausal_C = []
        anticausal_D = []
        for i in range(k):
            anticausal_C.append(
                causal_system.C[i] @ M[i] @ anticausal_system.A[i]
            )
            anticausal_D.append(np.zeros((
                causal_system.C[i].shape[0],
                anticausal_system.B[i].shape[1]
            )))
        result = result.add(
            RealizationStrict(
                causal=False,
                A=anticausal_system.A,
                B=anticausal_system.B,
                C=anticausal_C,
                D=anticausal_D
            )
        )
        # Causal pass-through part as a anticausal system
        pass_causal_A = []
        pass_causal_B = []
        pass_causal_C = []
        for i in range(k):
            pass_causal_A.append(np.zeros((0,0)))
            pass_causal_B.append(np.zeros((
                0,
                causal_system.D[i].shape[1]
            )))
            pass_causal_C.append(np.zeros((
                causal_system.D[i].shape[0],
                0
            )))
        R_causal_pass_as_anticausal = RealizationStrict(
            causal=False,
            A=pass_causal_A,
            B=pass_causal_B,
            C=pass_causal_C,
            D=causal_system.D
        )
        # Anticausal pass-through part as causal and anticausal system
        pass_anticausal_A = []
        pass_anticausal_B = []
        pass_anticausal_C = []
        for i in range(k):
            pass_anticausal_A.append(np.zeros((0,0)))
            pass_anticausal_B.append(np.zeros((
                0,
                anticausal_system.D[i].shape[1]
            )))
            pass_anticausal_C.append(np.zeros((
                anticausal_system.D[i].shape[0],
                0
            )))
        R_anticausal_pass_as_causal = RealizationStrict(
            causal=True,
            A=pass_anticausal_A,
            B=pass_anticausal_B,
            C=pass_anticausal_C,
            D=anticausal_system.D
        )
        R_anticausal_pass_as_anticausal = RealizationStrict(
            causal=False,
            A=pass_anticausal_A,
            B=pass_anticausal_B,
            C=pass_anticausal_C,
            D=anticausal_system.D
        )
        # Adding product of pass-through parts
        result = result.add(
            R_causal_pass_as_anticausal.mul(R_anticausal_pass_as_anticausal)
        )
        # Anticausal system without pass-through part
        pure_anticausal_D = []
        for i in range(k):
            pure_anticausal_D.append(np.zeros(anticausal_system.D[i].shape))
        R_pure_anticausal = RealizationStrict(
            causal=False,
            A=anticausal_system.A,
            B=anticausal_system.B,
            C=anticausal_system.C,
            D=pure_anticausal_D
        )
        # Adding product of causal pass-through system and pure anticausal system
        result = result.add(
            R_causal_pass_as_anticausal.mul(R_pure_anticausal)
        )
        # Causal system without pass-through part
        pure_causal_D = []
        for i in range(k):
            pure_causal_D.append(np.zeros(causal_system.D[i].shape))
        R_pure_causal = RealizationStrict(
            causal=True,
            A=causal_system.A,
            B=causal_system.B,
            C=causal_system.C,
            D=pure_causal_D
        )
        # Adding product of pure causal system and anticausal pass-through system
        result = result.add(
            R_pure_causal.mul(R_anticausal_pass_as_causal)
        )
        return result

    def _anticausal_times_causal(
        self,
        anticausal_system:RealizationStrict,
        causal_system:RealizationStrict):
        k = len(causal_system.A)
        M = [np.zeros((0,0))]*(k+1)
        for i in range(k-1,-1,-1):
            M[i] = anticausal_system.B[i] @ causal_system.C[i] \
                + anticausal_system.A[i] @ M[i+1] @ causal_system.A[i]
        # Compute strictly anticausal part of multiplication result
        anticausal_B = []
        anticausal_D = []
        for i in range(k):
            anticausal_B.append(
                anticausal_system.A[i] @ M[i+1] @ causal_system.B[i]
            )
            anticausal_D.append(
                anticausal_system.C[i] @ M[i+1] @ causal_system.B[i]
            )
        result = RealizationStrict(
            causal=False,
            A=anticausal_system.A,
            B=anticausal_B,
            C=anticausal_system.C,
            D=anticausal_D
        )
        # Compute strictly causal part of multiplication result
        causal_C = []
        causal_D = []
        for i in range(k):
            causal_C.append(
                anticausal_system.C[i] @ M[i+1] @ causal_system.A[i]
            )
            causal_D.append(np.zeros((
                anticausal_system.C[i].shape[0],
                causal_system.B[i].shape[1]
            )))
        result = result.add(RealizationStrict(
            causal=True,
            A=causal_system.A,
            B=causal_system.B,
            C=causal_C,
            D=causal_D
        ))
        # Causal pass-through part as anticausal system
        pass_causal_A = []
        pass_causal_B = []
        pass_causal_C = []
        for i in range(k):
            pass_causal_A.append(np.zeros((0,0)))
            pass_causal_B.append(np.zeros((
                0,
                causal_system.D[i].shape[1]
            )))
            pass_causal_C.append(np.zeros((
                causal_system.D[i].shape[0],
                0
            )))
        R_causal_pass_as_anticausal = RealizationStrict(
            causal=False,
            A=pass_causal_A,
            B=pass_causal_B,
            C=pass_causal_C,
            D=causal_system.D
        )
        # Anticausal pass-through part as causal and anticausal system
        pass_anticausal_A = []
        pass_anticausal_B = []
        pass_anticausal_C = []
        for i in range(k):
            pass_anticausal_A.append(np.zeros((0,0)))
            pass_anticausal_B.append(np.zeros((
                0,
                anticausal_system.D[i].shape[1]
            )))
            pass_anticausal_C.append(np.zeros((
                anticausal_system.D[i].shape[0],
                0
            )))
        R_anticausal_pass_as_causal = RealizationStrict(
            causal=True,
            A=pass_anticausal_A,
            B=pass_anticausal_B,
            C=pass_anticausal_C,
            D=anticausal_system.D
        )
        R_anticausal_pass_as_anticausal = RealizationStrict(
            causal=False,
            A=pass_anticausal_A,
            B=pass_anticausal_B,
            C=pass_anticausal_C,
            D=anticausal_system.D
        )
        # Adding product of pass-through parts
        result = result.add(
            R_anticausal_pass_as_anticausal.mul(R_causal_pass_as_anticausal)
        )
        # Anticausal system without pass-through part
        pure_anticausal_D = []
        for i in range(k):
            pure_anticausal_D.append(np.zeros(
                anticausal_system.D[i].shape
            ))
        R_pure_anticausal = RealizationStrict(
            causal=False,
            A=anticausal_system.A,
            B=anticausal_system.B,
            C=anticausal_system.C,
            D=pure_anticausal_D
        )
        # Adding product of pure anticausal and causal pass-through system
        result = result.add(
            R_pure_anticausal.mul(R_causal_pass_as_anticausal)
        )
        # Causal system without pass-through part
        pure_causal_D = []
        for i in range(k):
            pure_causal_D.append(np.zeros(
                causal_system.D[i].shape
            ))
        R_pure_causal = RealizationStrict(
            causal=True,
            A=causal_system.A,
            B=causal_system.B,
            C=causal_system.C,
            D=pure_causal_D
        )
        # Adding product of anticausal pass-through and pure causal system
        result = result.add(
            R_anticausal_pass_as_causal.mul(R_pure_causal)
        )
        return result
