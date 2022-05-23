import numpy as np
from tvsclib.canonical_form import CanonicalForm
from tvsclib.mixed_system import MixedSystem
from tvsclib.strict_system import StrictSystem
from tvsclib.identification import identify
import tvsclib.utils as utils
import tvsclib.math as math
from tempfile import TemporaryFile


def testUtils():
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

    #test storing and loading mixed systems
    file = TemporaryFile()
    system,sigmas = identify(matrix,dims_in,dims_out,compute_sigmas=True)
    utils.save_system(system,file,sigmas=sigmas)
    file.seek(0)
    system_load,sigmas_load = utils.load_system(file,load_sigmas=True)

    assert np.allclose(system.to_matrix(),system_load.to_matrix()),"loaded system incorrect"

    for sig_a,sig_b in zip(sigmas,sigmas_load):
        for s_a,s_b in zip(sig_a,sig_b):
            assert np.all(s_a==s_b), "loaded sigmas not equal"
    #TODO tests for other cases
