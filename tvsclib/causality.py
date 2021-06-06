from enum import Enum

class Causality(Enum):
    """ Causality of a state space system. """
    CAUSAL = 1
    ANTICAUSAL = 2
    MIXED = 3