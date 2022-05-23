import numpy as np
from tvsclib.canonical_form import CanonicalForm
from tvsclib.mixed_system import MixedSystem
from tvsclib.strict_system import StrictSystem
import tvsclib.identification as ident
import tvsclib.utils as utils
import tvsclib.math as math

def testSystemIdentificationSVD():
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

    matrix_causal = matrix.copy()
    for i in range(len(dims_in)):
        matrix_causal[:np.sum(dims_out[:i+1]),np.sum(dims_in[:i+1]):]=0
    print(matrix_causal)

    for canonical_form in (CanonicalForm.OUTPUT,CanonicalForm.INPUT,CanonicalForm.BALANCED):
        system = ident.identify_causal(matrix,dims_in,dims_out,canonical_form=canonical_form)

        correct,rep = utils.check_dims(system,text_output=False,return_report=True)
        assert correct, rep

        assert np.allclose(system.to_matrix(),matrix_causal), "Identifiaction incorrect"

        if canonical_form==CanonicalForm.INPUT:
            assert system.is_input_normal(),"System not input normal"
        elif canonical_form==CanonicalForm.OUTPUT:
            assert system.is_output_normal(),"System not output normal"
        elif canonical_form==CanonicalForm.BALANCED:
            assert system.is_balanced(),"System not balanced"

    matrix_anticausal = matrix-matrix_causal

    for canonical_form in (CanonicalForm.OUTPUT,CanonicalForm.INPUT,CanonicalForm.BALANCED):
        system = ident.identify_anticausal(matrix,dims_in,dims_out,canonical_form=canonical_form)


        correct,rep = utils.check_dims(system,text_output=False,return_report=True)
        assert correct, rep

        assert np.allclose(system.to_matrix(),matrix_anticausal), "Identifiaction incorrect"

        if canonical_form==CanonicalForm.INPUT:
            assert system.is_input_normal(),"System not input normal"
        elif canonical_form==CanonicalForm.OUTPUT:
            assert system.is_output_normal(),"System not output normal"
        elif canonical_form==CanonicalForm.BALANCED:
            assert system.is_balanced(),"System not balanced"

    system,(sigmas_causal,sigmas_anticausal) = ident.identify(matrix,dims_in,dims_out,compute_sigmas=True)
    correct,rep = utils.check_dims(system,text_output=False,return_report=True)
    assert correct, rep

    assert np.allclose(system.to_matrix(),matrix), "Identifiaction incorrect"

    (sigmas_causal_refer,sigmas_anticausal_refer) = math.extract_sigmas(matrix, dims_in,dims_out)
    for i in range(len(sigmas_causal_refer)):
        sig_causal = np.zeros_like(sigmas_causal_refer[i])
        sig_anticausal = np.zeros_like(sigmas_anticausal_refer[i])
        sig_causal[:len(sigmas_causal[i])]=sigmas_causal[i]
        sig_anticausal[:len(sigmas_anticausal[i])]=sigmas_anticausal[i]
        assert np.allclose(sig_causal,sigmas_causal_refer[i]),\
                "Causal sigmas do not match"+str(i)+str(sigmas_causal_c[i])+str(sigmas_causal_refer[i])
        assert np.allclose(sig_anticausal,sigmas_anticausal_refer[i]),\
                "Anticausal sigmas do not match"+str(i)+str(sigmas_anticausal_c[i])+str(sigmas_anticausal_refer[i])
