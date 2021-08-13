# %%
import os
import warnings
from datetime import datetime
from numpy.lib.index_tricks import nd_grid
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import OAS, LedoitWolf
from sklearn.model_selection import StratifiedKFold
from tvsclib.strict_system import StrictSystem
from tvsclib.toeplitz_operator import ToeplitzOperator
from tvsclib.system_identification_svd import SystemIdentificationSVD
from tvsclib.transformations.reduction import Reduction

def load_UCR2018(base_dir, dataset_name):
    datasets = {}
    for what in ["TRAIN", "TEST"]:
        filename = f"{base_dir}/{dataset_name}/{dataset_name}_{what}.tsv"
        data = np.loadtxt(filename)
        x = data[:,1:]
        y = data[:,0]
        datasets[what] = {"X":x, "Y":y}
    return datasets

# %%
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

def band_chol_cv(X:np.ndarray, n_splits:int = 10, verbose:bool = False):
    n = X.shape[0]
    p = X.shape[1]

    n_tr = int(np.round(n*(1-1/np.log(n))))
    n_va = n - n_tr

    k_vec = np.arange(0,min(p-1,n-2))
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
            if verbose: print(f"Finished k={k_vec[i]} in split {ns}")
        
        if verbose: print(f"Finished split {ns}")
    
    cv_err = cv_loss.sum(axis=1)
    best_k = k_vec[np.argmin(cv_err)]
    if verbose: print(f"Best k={best_k}")
    sigma = band_chol(X, best_k)

    return (sigma, best_k)

def band_sample_cv(X:np.ndarray, n_splits:int = 10, verbose:bool = False):
    n = X.shape[0]
    p = X.shape[1]

    n_tr = int(np.round(n*(1-1/np.log(n))))
    n_va = n - n_tr

    k_vec = np.arange(0,min(p-1,n-2))
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
            if verbose: print(f"Finished k={k_vec[i]} in split {ns}")
        
        if verbose: print(f"Finished split {ns}")
    
    cv_err = cv_loss.sum(axis=1)
    best_k = k_vec[np.argmin(cv_err)]
    if verbose: print(f"Best k={best_k}")
    X_zero_mean = X - X.mean(axis=0)
    sigma = X_zero_mean.transpose() @ X_zero_mean
    sigma = np.triu(sigma, -best_k) - np.triu(sigma, best_k+1)
    return (sigma, best_k)

# %%
def ma_process(theta:np.ndarray, n:int, m:int):
    Y = np.zeros((n,m))
    for n_idx in range(n):
        err = np.random.randn(m + len(theta) - 1)
        y = np.convolve(err, theta, mode='valid')
        Y[n_idx,:] = y
    C = np.zeros((m,m))
    for tau in range(len(theta)):
        gamma = 0
        for i in range(len(theta) - tau):
            gamma = gamma + theta[i]*theta[tau+i]
        C = C + np.diag([1]*(m-np.abs(tau)),tau)*gamma
        if tau != 0:
            C = C + np.diag([1]*(m-np.abs(tau)),-tau)*gamma
    return (Y,C)

def tvarma_process(A:np.ndarray, B:np.ndarray, n:int):
    d = B.shape[1]
    w = np.random.randn(d,n)
    Y = np.linalg.inv(np.eye(d) - A)
    X = (Y@B@w).transpose()
    C = Y@B@B.transpose()@Y.transpose()
    return (X,C)

# %%
def make_sample_cov(X:np.ndarray):
    X_zero_mean = X - X.mean(axis=0)
    sample_cov = X_zero_mean.transpose() @ X_zero_mean
    return sample_cov, {"estimator": "Sample"}

def make_sample_band_cov(X:np.ndarray):
    sample_band_cov, bw = band_sample_cv(X,verbose=True)
    return sample_band_cov, {"estimator": "Sample band", "bw": bw}

def make_chol_band_cov(X:np.ndarray):
    chol_band_cov, bw = band_chol_cv(X, verbose=True)
    return chol_band_cov, {"estimator": "Cholesky band", "bw": bw}

def make_lw_cov(X:np.ndarray):
    lw_cov = LedoitWolf().fit(X).covariance_
    return lw_cov, {"estimator": "Ledoit Wolf"}

def make_ss_approx(C_est:np.ndarray, info:dict, epsilon:float, bw:int):
    input_dims = [ *([1]*C_est.shape[0]), *([0]*(bw+1)) ]
    output_dims = [ *([0]*(bw+1)), *([1]*C_est.shape[0]) ]
    T = ToeplitzOperator(C_est, input_dims, output_dims)
    Sid = SystemIdentificationSVD(T, epsilon=epsilon)
    Sys = StrictSystem(causal=True, system_identification=Sid)
    To,V = Sys.outer_inner_factorization()
    To_min = Reduction().apply(To)

    d = To_min.dims_state
    d_mean = np.mean(d)
    print(f"d_mean: {d_mean}")
    C_rec = To_min.to_matrix() @ V.to_matrix()
    C_rec_as_dict = dict(zip(
        [f"\\hat{{c}}_{{ss}}({idx})" for idx in range(1, C_rec.size+1)],
        C_rec.astype("float32").flatten()
    ))
    d_dict = dict(zip(
        [f"d({idx})" for idx in range(1, len(d)+1)],
        d
    ))
    info = {
        **info, 
        "\\epsilon": epsilon, 
        "estimator": info["estimator"] + " + SS",
        f"\\bar{{d}}": d_mean, 
        **d_dict,
        **C_rec_as_dict }

    return C_rec, info, To_min

# %%
def calc_eigenspace_agreement(eigv_true, eigv_est):
    eigv_true = np.sort(eigv_true.copy())[::-1]
    eigv_est = np.sort(eigv_est.copy())[::-1]
    K = np.zeros(len(eigv_true)).astype(complex)
    for q in range(len(eigv_true)):
        for i in range(q):
            for j in range(q):
                K[q] = K[q] + (np.conj(eigv_est[i]) * eigv_true[j])**2
    return K

def calc_statistics(C_true, C_est):
    eigvals_true = np.linalg.eigvals(C_true)
    eigvals_est = np.linalg.eigvals(C_est)
    eigspace_agreement = calc_eigenspace_agreement(eigvals_true, eigvals_est)
    
    pdf = np.all(np.real(eigvals_est) > 0)
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
    C_true_as_dict = dict(zip(
        [f"c({idx})" for idx in range(1, C_true.size+1)],
        C_true.astype("float32").flatten()
    ))
    C_est_as_dict = dict(zip(
        [f"\\hat{{c}}({idx})" for idx in range(1, C_est.size+1)],
        C_est.astype("float32").flatten()
    ))

    return { "mse": mse, "pdf": pdf, 
        **eigvals_true_dict, **eigvals_est_dict, **eigspace_agreement_dict,
        **C_true_as_dict, **C_est_as_dict }
  
#%%
from datetime import datetime
import pandas as pd

results_folder = "./results"
timestamp = datetime.today().strftime('%Y%m%d_%H%M%S%f')[:-3]

def save_statistics(prefix:str, stats):
    result_filename = f"{prefix}_{timestamp}"
    df = pd.DataFrame(stats)
    df.to_pickle(f"{results_folder}/{result_filename}.pkl")
    df.to_csv(f"{results_folder}/{result_filename}.csv")
    #df.to_excel(f"{results_folder}/{result_filename}.xlsx")  
#%%

np.random.seed(hash(timestamp) & (2*32 - 1))

process_lens = [
    200, 
    100, 
    30
]
sample_sizes = [50, 100, 150, 200, 250, 300, 350, 400]
epsilon_values = [0.01, 0.05, 0.1]
processes = {
    "tvARMA(2,2)": lambda n,m: tvarma_process(
        np.diag(0.6*(1 + 0.1*np.random.randn(m-1)),-1) + np.diag(0.3*(1 + 0.1*np.random.randn(m-2)),-2),
        np.diag([1] * m) + np.diag([1] * (m-1),-1), n),
    "MA(4)": lambda n,m: ma_process(theta=[1, 5/10, 4/10, 3/10, 2/10], n=n, m=m)
}
estimators = [
    make_chol_band_cov,
    make_sample_band_cov,
    make_sample_cov,
    make_lw_cov,
]
no_draws = 10
sim_results = []
iter = 0
for draw_idx in range(no_draws):
    print(f"Draw: {draw_idx+1}")
    for proc_len in process_lens:
        print(f"Process length: {proc_len}")
        for proc_name, proc in processes.items():
            X,C_true = proc(np.max(sample_sizes), proc_len)
            for sample_size in sample_sizes:
                print(f"Sample size: {sample_size}")
                X_sub = X[-sample_size:,:].copy()
                for estimator in [*estimators, lambda _:(C_true,{"estimator": "True"})]:
                    #if iter < 1699:
                    #    iter = iter + 1
                    #    continue
                    C_est, info = estimator(X_sub)
                    print(info)
                    stats = calc_statistics(C_true, C_est)
                    sim_results.append({
                        "Process": proc_name,
                        "n": sample_size,
                        "p": proc_len,
                        **info,
                        **stats
                    })
                    if "bw" in info:
                        # Compute SS approximation
                        for epsilon in epsilon_values:
                            print(f"Epsilon: {epsilon}")
                            C_rec,info_ss,_ = make_ss_approx(C_est, info, epsilon, info["bw"])
                            stats_ss = calc_statistics(C_true, C_rec)
                            sim_results.append({
                                "Process": proc_name,
                                "n": sample_size,
                                "p": proc_len,
                                **info,
                                **stats,
                                **info_ss,
                                **stats_ss
                            })
                    save_statistics(f"sim-{iter}", sim_results)
                    iter = iter + 1
                    sim_results = []

# %%
import glob

selected = [
    "Process", 
    "estimator", 
    "p", 
    "n", 
    "mse", 
    "\\epsilon", 
    f"\\bar{{d}}", 
    "bw" ]
dataframes = []

for file in glob.glob(f"{results_folder}/*.pkl"):
    df = pd.read_pickle(file)
    dataframes.append(df[df.columns.intersection(set(selected))])

df_statistics = pd.concat(dataframes,ignore_index=True)

# %%
import seaborn as sns

p_values = df_statistics["p"].unique()
process_values = df_statistics["Process"].unique()

plt.style.use(['science','ieee', 'high-vis'])

for process in process_values:
    for p in p_values:
        df = df_statistics[
            (df_statistics["Process"] == process) 
            & (df_statistics["p"] == p)
            & df_statistics["estimator"].isin(["Sample", "Ledoit Wolf", "Cholesky band", "Sample band"])
            ].copy()
        df.loc[:,f"log$_{{10}}$(mse)"] = np.log10(df["mse"])
        g = sns.lineplot(data=df, x="n", y=f"log$_{{10}}$(mse)", hue="estimator", ci=None, legend=True)
        h,l = g.get_legend_handles_labels()
        g.legend(h,l,bbox_to_anchor=(0.5, -0.15),
            loc='upper center', ncol=2)
        plt.title(f"{process}, p={p}")
        plt.show()
        df = df_statistics[
            (df_statistics["Process"] == process) 
            & (df_statistics["p"] == p)
            & df_statistics["estimator"].isin(["Cholesky band + SS"])
            ].copy()
        df.loc[:,f"log$_{{10}}$(mse)"] = np.log10(df["mse"])
        df.loc[:,"estimator"] = df.apply(lambda row: "Approx. $\\epsilon=" + str(row["\\epsilon"]) + "$", axis=1)
        sns.lineplot(data=df, x="n", y=f"log$_{{10}}$(mse)", hue="estimator", ci=None, legend=True)
        df = df_statistics[
            (df_statistics["Process"] == process) 
            & (df_statistics["p"] == p)
            & df_statistics["estimator"].isin(["Ledoit Wolf", "Cholesky band"])
            ].copy()
        df.loc[:,f"log$_{{10}}$(mse)"] = np.log10(df["mse"])
        g = sns.lineplot(data=df, x="n", y=f"log$_{{10}}$(mse)", hue="estimator", ci=None, legend=True)
        h,l = g.get_legend_handles_labels()
        g.legend(
            h, l, 
            bbox_to_anchor=(0.5, -0.15),loc='upper center', ncol=2)
        plt.title(f"{process}, p={p}")
        plt.show()
        

# %%
dataset_folder = "./.local/UCRArchive_2018"
dataset_names = [
    #"ElectricDevices", # 96 Features
    #"StarLightCurves", # 1024 Features
    #"Wafer", # 152 Features
    #"ECG200", # 96 Features
    #"PowerCons", # 144 Features
    #"FreezerRegularTrain", # 301 Features
    #"Strawberry", # 235 Features
    #"FacesUCR", # 131 Features
]

for dataset_name in dataset_names:
    dataset = load_UCR2018(dataset_folder, dataset_name)
    X = np.vstack([
        dataset["TRAIN"]["X"], 
        dataset["TEST"]["X"]
    ])
    Y = np.hstack([
        dataset["TRAIN"]["Y"], 
        dataset["TEST"]["Y"]
    ])
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
    for train_idx,test_idx in kfold.split(X,Y):
        X_train, X_test = X[train_idx,:], X[test_idx,:]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

# %%
