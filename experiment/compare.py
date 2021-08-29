import numpy as np

def eigenspace_agreement(eigv_true, eigvec_true, eigv_est, eigvec_est):
    order_true = np.argsort(eigv_true)[::-1]
    order_est = np.argsort(eigv_est)[::-1]
    eigv_true = eigv_true[order_true]
    eigv_est = eigv_est[order_est]
    eigvec_true = eigvec_true[:,order_true]
    eigvec_est = eigvec_est[:,order_est]
    K = np.zeros(len(eigv_true)).astype(complex)
    for q in range(len(eigv_true)):
        for i in range(q):
            for j in range(q):
                K[q] = K[q] + (eigvec_est[:,i] @ eigvec_true[:,j])**2
    return K

def compare_cov(C_true, C_est):
    eigvals_true,eigvecs_true = np.linalg.eigh(C_true)
    eigvals_est,eigvecs_est = np.linalg.eigh(C_est)
    eigspace_agreement = eigenspace_agreement(
        eigvals_true, eigvecs_true, eigvals_est, eigvecs_est)
    
    pdf = np.all(eigvals_est > 1e-12) \
        and np.allclose(C_est, C_est.transpose())
    mse = np.sum((C_true - C_est)**2) / C_true.shape[0] / C_true.shape[1]

    eigvals_true_dict = dict(zip(
        [f"\\lambda({idx})" for idx in range(1, len(eigvals_true)+1)],
        eigvals_true
    ))
    eigvals_est_dict = dict(zip(
        [f"\\hat{{\\lambda}}({idx})" for idx in range(1, len(eigvals_est)+1)],
        eigvals_est
    ))
    eigspace_agreement_dict = dict(zip(
        [f"K({idx})" for idx in range(1, len(eigspace_agreement)+1)],
        eigspace_agreement
    ))
    
    return { 
        "mse": mse, 
        "pdf": pdf, 
        **eigvals_true_dict, 
        **eigvals_est_dict, 
        **eigspace_agreement_dict }