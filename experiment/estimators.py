import warnings
import numpy as np
from sklearn.covariance import LedoitWolf
from tvsclib.strict_system import StrictSystem
from tvsclib.toeplitz_operator import ToeplitzOperator
from tvsclib.system_identification_svd import SystemIdentificationSVD

def band_chol(X:np.ndarray, k:int):
    # Centering X
    X = X - X.mean(axis=0)

    n = X.shape[0]
    p = X.shape[1]
    if k > p-1:
        warnings.warn("k is greater than p-1, set k = p-1")
        k = p-1
    chol = np.eye(p)
    resid = X[:,0].reshape((-1,1))
    sigma2 = [np.mean(resid**2)]
    for j in range(1,p):
        if k >= 1:
            keep = range(max(j-k, 0),j)
            newy = X[:,j]
            newx = resid[:,keep]

            if len(keep) < n-1:
                xtx = newx.transpose() @ newx
                xty = newx.transpose() @ newy
                q,r = np.linalg.qr(xtx)
                phi_new = np.linalg.inv(r) @ q.transpose() @ xty
            else:
                u,s,vh = np.linalg.svd(newx)
                u = u[:,0:n-1]
                vh = vh[0:n-1,:]
                w = np.diag(1/s[0:n-1])
                phi_new = vh.transpose() @ w @ u.transpose() @ newy

            thisresid = newy - newx@phi_new
            sigma_new = 0
            if len(keep) < n-1:
                sigma_new = np.mean(thisresid**2)
            chol[j,keep] = phi_new
            resid = np.hstack([resid, thisresid.reshape((resid.shape[0],-1))])
            sigma2.append(sigma_new)
        else:
            sigma2.append(np.mean(X[:,j]**2))
    return chol @ np.diag(sigma2) @ chol.transpose()

def band_chol_cv(X:np.ndarray, n_splits:int = 10, verbose:bool = False, k_max:int = -1):
    n = X.shape[0]
    p = X.shape[1]

    if k_max == -1:
        k_max = p

    n_tr = int(np.round(n*(1-1/np.log(n))))
    n_va = n - n_tr

    k_vec = np.arange(0,min(p-1,n-2, k_max))
    cv_loss = np.zeros((len(k_vec), n_splits))

    for ns in range(n_splits):
        ind = np.arange(0,n)
        np.random.shuffle(ind)
        ind_tr = ind[0:n_tr]
        ind_va = ind[n_tr:]
        x_tr = X[ind_tr,:]
        x_va = X[ind_va,:]
        x_va = x_va - x_va.mean(axis=0)
        s_va = x_va.transpose() @ x_va / (n_va - 1)
                
        for i in range(len(k_vec)):
            sigma = band_chol(x_tr, k_vec[i])
            cv_loss[i,ns] = np.sum((sigma - s_va)**2)        
        if verbose: print(f"Finished split {ns}")
    
    cv_err = cv_loss.sum(axis=1)
    best_k = k_vec[np.argmin(cv_err)]
    if verbose: print(f"Best k={best_k}")
    sigma = band_chol(X, best_k)

    return (sigma, best_k)

def band_sample_cv(X:np.ndarray, n_splits:int = 10, verbose:bool = False, k_max:int = -1):
    n = X.shape[0]
    p = X.shape[1]

    if k_max == -1:
        k_max = p

    n_tr = int(np.round(n*(1-1/np.log(n))))
    n_va = n - n_tr

    k_vec = np.arange(0,min(p-1,n-2,k_max))
    cv_loss = np.zeros((len(k_vec), n_splits))

    for ns in range(n_splits):
        ind = np.arange(0,n)
        np.random.shuffle(ind)
        ind_tr = ind[0:n_tr]
        ind_va = ind[n_tr:]
        x_tr = X[ind_tr,:]
        x_tr = x_tr - x_tr.mean(axis=0)
        x_va = X[ind_va,:]
        x_va = x_va - x_va.mean(axis=0)
        s_va = x_va.transpose() @ x_va / (n_va - 1)
                
        for i in range(len(k_vec)):
            sigma = x_tr.transpose() @ x_tr / (n_tr - 1)
            sigma = np.triu(sigma, -k_vec[i]) - np.triu(sigma, k_vec[i]+1)
            cv_loss[i,ns] = np.sum((sigma - s_va)**2)        
        if verbose: print(f"Finished split {ns}")
    
    cv_err = cv_loss.sum(axis=1)
    best_k = k_vec[np.argmin(cv_err)]
    if verbose: print(f"Best k={best_k}")
    X_zero_mean = X - X.mean(axis=0)
    sigma = X_zero_mean.transpose() @ X_zero_mean
    sigma = np.triu(sigma, -best_k) - np.triu(sigma, best_k+1)
    return (sigma, best_k)

def make_sample_cov(X:np.ndarray):
    X_zero_mean = X - X.mean(axis=0)
    p = X.shape[1]
    params = p*(p+1)/2 # Assuming positive definitness
    sample_cov = X_zero_mean.transpose() @ X_zero_mean
    flops = p**2
    return sample_cov, {
        "flops": flops,
        "speedup": 1.0,
        "params": params,
        "savings": 1.0,
        "estimator": "Sample" }

def make_sample_diag_cov(X:np.ndarray):
    X_zero_mean = X - X.mean(axis=0)
    sample_cov = X_zero_mean.transpose() @ X_zero_mean
    diag_cov = np.diag(np.diag(sample_cov))
    p = diag_cov.shape[0]
    params = p
    savings = p*(p+1)/2 / params
    flops = p
    speedup = p**2 / flops
    return diag_cov, {
        "params": params,
        "savings": savings,
        "flops": flops,
        "speedup": speedup,
        "estimator": "Sample diag." }

def make_identity_cov(X:np.ndarray):
    return np.identity(X.shape[1]), {
        "params": 0,
        "savings": float("inf"),
        "flops": 0,
        "speedup": float("inf"),
        "estimator": "Identity"
    }

def make_sample_band_cov(X:np.ndarray):
    p = X.shape[1]
    sample_band_cov, bw = band_sample_cv(X,verbose=True, k_max=int(np.ceil(np.sqrt(p))))
    params = p*(p+1)/2 - (p-bw-1)*(p-bw)/2
    savings = p*(p+1)/2 / params
    flops = (bw+1)**2 + (p-bw-1)*(2*bw+1)
    speedup = p**2 / flops
    return sample_band_cov, {
        "params": params,
        "savings": savings,
        "flops": flops,
        "speedup": speedup,
        "estimator": "Sample band", 
        "bw": bw }

def make_chol_band_cov(X:np.ndarray):
    p = X.shape[1]
    chol_band_cov, bw = band_chol_cv(X, verbose=True, k_max=int(np.ceil(np.sqrt(p))))
    params = p*(p+1)/2 - (p-bw-1)*(p-bw)/2
    savings = p*(p+1)/2 / params
    flops = (bw+1)**2 + (p-bw-1)*(2*bw+1)
    speedup = p**2 / flops
    return chol_band_cov, {
        "params": params,
        "savings": savings,
        "flops": flops,
        "speedup": speedup,
        "estimator": "Cholesky band", 
        "bw": bw }

def make_lw_cov(X:np.ndarray):
    lw_cov = LedoitWolf().fit(X).covariance_
    p = X.shape[1]
    params = p*(p+1)/2
    flops = p**2
    return lw_cov, {
        "params": params,
        "speedup": 1.0,
        "flops": flops,
        "savings": 1.0,
        "estimator": "Ledoit Wolf" }

def make_ss_approx(make_cov, X:np.ndarray, epsilon:float, as_matrix:bool=False):
    C_est, C_info = make_cov(X)
    p = C_est.shape[1]
    input_dims = [1]*C_est.shape[0]
    output_dims = [1]*C_est.shape[0]
    try:
        C_est_chol = np.linalg.cholesky(C_est)
    except:
        raise AttributeError("Matrix not PDF")

    T = ToeplitzOperator(C_est_chol, input_dims, output_dims)
    Sid = SystemIdentificationSVD(T, epsilon=epsilon)
    Sys = StrictSystem(causal=True, system_identification=Sid)

    d = np.array(Sys.dims_state)
    d_mean = np.mean(d)
    print(f"d_mean: {d_mean}")

    flops = 2*np.sum(d**2) + 3*np.sum(d) + 1
    print(f"flops: {flops}")

    speedup = (p**2) / flops
    print(f"speedup: {speedup}")

    params = np.sum(d**2) + 2*np.sum(d) + len(d)
    print(f"params: {params}")

    savings = p*(p+1)/2 / params
    print(f"savings: {savings}")

    info = { 
        **C_info,
        "estimator": C_info["estimator"] + " + SS",
        "params": params,
        "savings": savings,
        "flops": flops,
        "speedup": speedup,
        "\\epsilon": epsilon, 
        f"\\bar{{d}}": d_mean, 
        #**d_dict 
        }

    if as_matrix:
        mat_rec = Sys.to_matrix()
        return mat_rec @ mat_rec.transpose(),info
    return Sys,info