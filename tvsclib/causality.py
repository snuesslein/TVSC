""" Definition of the causality enum. """
from enum import Enum

class Causality(Enum):
    """ A system can either be causal, anticausal or both. """
    CAUSAL = 1,
    ANTICAUSAL = 2,
    MIXED = 3