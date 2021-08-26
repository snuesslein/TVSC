# %% Processes
from typing import Callable
import numpy as np
from numpy.core.numeric import identity
from seaborn.palettes import color_palette

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

# %% Covariance estimators
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
    return sample_cov, {
        "params": params,
        "estimator": "Sample" }

def make_sample_diag_cov(X:np.ndarray):
    X_zero_mean = X - X.mean(axis=0)
    sample_cov = X_zero_mean.transpose() @ X_zero_mean
    diag_cov = np.diag(np.diag(sample_cov))
    params = diag_cov.shape[0]
    return diag_cov, {
        "params": params,
        "estimator": "Sample diag." }

def make_identity_cov(X:np.ndarray):
    return np.identity(X.shape[1]), {
        "params": 0,
        "estimator": "Identity"
    }

def make_sample_band_cov(X:np.ndarray):
    p = X.shape[1]
    sample_band_cov, bw = band_sample_cv(X,verbose=True, k_max=int(np.ceil(np.sqrt(p))))
    params = p*(p+1)/2 - (p-bw-1)*(p-bw)/2
    return sample_band_cov, {
        "params": params,
        "estimator": "Sample band", 
        "bw": bw }

def make_chol_band_cov(X:np.ndarray):
    p = X.shape[1]
    chol_band_cov, bw = band_chol_cv(X, verbose=True, k_max=int(np.ceil(np.sqrt(p))))
    params = p*(p+1)/2 - (p-bw-1)*(p-bw)/2
    return chol_band_cov, {
        "params": params,
        "estimator": "Cholesky band", 
        "bw": bw }

def make_lw_cov(X:np.ndarray):
    lw_cov = LedoitWolf().fit(X).covariance_
    p = X.shape[1]
    params = p*(p+1)/2
    return lw_cov, {
        "params": params,
        "estimator": "Ledoit Wolf" }

def make_ss_approx(make_cov:Callable, X:np.ndarray, epsilon:float, as_matrix:bool=False):
    C_est, C_info = make_cov(X)
    input_dims = [1]*C_est.shape[0]
    output_dims = [1]*C_est.shape[0]
    try:
        C_est_chol = np.linalg.cholesky(C_est)
    except:
        raise AttributeError("Matrix not PDF")

    T = ToeplitzOperator(C_est_chol, input_dims, output_dims)
    Sid = SystemIdentificationSVD(T, epsilon=epsilon)
    Sys = StrictSystem(causal=True, system_identification=Sid)

    d = Sys.dims_state
    d_mean = np.mean(d)
    print(f"d_mean: {d_mean}")
    d_dict = dict(zip(
        [f"d({idx})" for idx in range(1, len(d)+1)],
        d
    ))
    params = np.sum(np.array(Sys.dims_state)**2) \
        + np.sum(2*np.array(Sys.dims_state)) + len(Sys.stages)
    info = { 
        **C_info,
        "estimator": C_info["estimator"] + " + SS",
        "params": params,
        "\\epsilon": epsilon, 
        f"\\bar{{d}}": d_mean, 
        **d_dict }

    if as_matrix:
        mat_rec = Sys.to_matrix()
        return mat_rec @ mat_rec.transpose(),info
    return Sys,info

# %% Evaluation of covariances
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

# %% Data handling
import glob
import pandas as pd
import numpy as np

def save_statistics(prefix:str, timestamp:str, results_folder:str, stats):
    result_filename = f"{prefix}_{timestamp}"
    df = pd.DataFrame(stats)
    df.to_pickle(f"{results_folder}/{result_filename}.pkl")
    df.to_csv(f"{results_folder}/{result_filename}.csv")
    df.to_excel(f"{results_folder}/{result_filename}.xlsx")

def load_statistics(prefix:str, results_folder:str):
    dataframes = []
    for file in glob.glob(f"{results_folder}/{prefix}_*.pkl"):
        df = pd.read_pickle(file)
        dataframes.append(df)
    return pd.concat(dataframes,ignore_index=True)

def load_UCR2018(base_dir, dataset_name):
    datasets = {}
    for what in ["TRAIN", "TEST"]:
        filename = f"{base_dir}/{dataset_name}/{dataset_name}_{what}.tsv"
        data = np.loadtxt(filename)
        x = data[:,1:]
        y = data[:,0]
        datasets[what] = {"X":x, "Y":y}
    return datasets

# %% LDA Classifiers
from typing import Callable
import numpy as np

class ClassifierBase:
    def __init__(self, name:str):
        self.labels = []
        self.name = name
    
    def fit(self, x_train, y_train):
        raise NotImplementedError()

    def distances(self, x_test):
        raise NotImplementedError()

    def predict(self, x_test):
        distances = self.distances(x_test)
        return self.labels[np.argmin(distances, axis=0)]

class MatrixClassifier(ClassifierBase):
    def __init__(self, name, make_cov:Callable[[np.ndarray], any]):
        super().__init__(name)
        self.make_cov = make_cov
        
    def fit(self, x_train, y_train):
        self.labels = np.unique(y_train)
        cov,info = self.make_cov(x_train)
        self.cov = cov
        self.inv_cov = np.linalg.inv(cov)
        self.info = { **info, "clf": self.name }
        self.mean_vectors = {}
        for label in self.labels:
            data = x_train[y_train == label, :]
            self.mean_vectors[label] = data.mean(axis=0)
    
    def distances(self, x_test):
        distances = []
        for label in self.labels:
            x_0 = x_test - self.mean_vectors[label]
            w = []
            for i in range(x_0.shape[0]):
                x_i = x_0[i,:]
                w.append(np.linalg.norm(x_i @ self.inv_cov \
                    @ x_i.transpose()))
            distances.append(w)
        return np.vstack(distances)

class StateSpaceClassifier(ClassifierBase):
    def __init__(self, name, make_sys:Callable[[np.ndarray], any]):
        super().__init__(name)
        self.make_sys = make_sys
    
    def fit(self, x_train, y_train):
        self.labels = np.unique(y_train)
        Sys,info = self.make_sys(x_train)
        self.info = { **info, "clf": self.name }
        self.inv_cov_ss = Sys.arrow_reversal()
        self.mean_vectors = {}
        for label in self.labels:
            data = x_train[y_train == label, :]
            self.mean_vectors[label] = data.mean(axis=0)
        
    def distances(self, x_test):
        distances = []
        for label in self.labels:
            x_0 = x_test - self.mean_vectors[label]
            w = []
            for i in range(x_0.shape[0]):
                x_i = x_0[i,:].reshape((-1,1))
                _,y_ss = self.inv_cov_ss.compute(x_i)
                w.append(np.linalg.norm(y_ss))
            distances.append(w)
        return np.vstack(distances)

# %% Classification performance measures
def confusion_matrix(y_pred, y_true, labels):
    n_classes = len(labels)
    cm = np.zeros([n_classes]*2, dtype=np.int32)
    i = 0
    for label_pred in labels:
        j = 0
        for label_true in labels:
            cm[i,j] = np.sum(
                (y_pred.flatten() == label_pred).astype("int32")\
                    * (y_true.flatten() == label_true))
            j = j+1
        i = i+1
    return cm

def accuracy(cm):
    return np.sum(np.diag(cm)) / np.sum(cm)

def precision(cm):
    prec = 0
    N = cm.shape[0]
    for i in range(N):
        prec = prec + cm[i,i] / np.sum(cm[:,i])
    prec = prec / N
    return prec

def recall(cm):
    reca = 0
    N = cm.shape[0]
    for i in range(N):
        reca = reca + cm[i,i] / np.sum(cm[i,:])
    reca = reca / N
    return reca

def f1(cm):
    prec = precision(cm)
    reca = recall(cm)
    score = 2*prec*reca / (prec + reca)
    return score

# %% Experiment with simulated data
from datetime import datetime
import numpy as np

generate_data = False
results_folder = "./results"
timestamp = datetime.today().strftime('%Y%m%d_%H%M%S%f')[:-3]

process_lens = [ 
    200,
    100,
    50
]
sample_sizes = [50, 100, 150, 200]
epsilon_values = [
#    0.1,
#    0.2,
#    0.3,
#    0.4,
    0.5
]

processes = {
    "tvARMA(2,2)": lambda n,m: tvarma_process(
        np.diag(0.6*(1 + 0.1*np.random.randn(m-1)),-1) + np.diag(0.3*(1 + 0.1*np.random.randn(m-2)),-2),
        np.diag([1] * m) + np.diag([1] * (m-1),-1), n),
    "MA(4)": lambda n,m: ma_process(theta=[1, 5/10, 4/10, 3/10, 2/10], n=n, m=m)
}
estimators = [
    make_sample_cov,
    make_sample_band_cov,
    make_chol_band_cov,
    make_lw_cov,
    *[(lambda X,epsilon=epsilon: make_ss_approx(make_lw_cov, X, epsilon, True)) for epsilon in epsilon_values],
    *[(lambda X,epsilon=epsilon: make_ss_approx(make_sample_cov, X, epsilon, True)) for epsilon in epsilon_values],
    *[(lambda X,epsilon=epsilon: make_ss_approx(make_chol_band_cov, X, epsilon, True)) for epsilon in epsilon_values]
]
no_draws = 10
iter = 0
for draw_idx in range(no_draws * generate_data):
    print(f"Draw: {draw_idx}")
    for proc_len in process_lens:
        print(f"Process length: {proc_len}")
        for proc_name, proc in processes.items():
            print(f"Process: {proc_name}")
            X,C_true = proc(np.max(sample_sizes), proc_len)
            for sample_size in sample_sizes:
                print(f"Sample size: {sample_size}")
                X_sub = X[-sample_size:,:].copy()
                make_true = lambda _: (C_true, { 
                    "estimator": "True", 
                    "params": proc_len*(proc_len+1)/2 }) 
                estimators_ext = [
                    make_true,
                    *[(lambda X,epsilon=epsilon: make_ss_approx(make_true, X, epsilon, True)) for epsilon in epsilon_values],
                    *estimators
                ]
                for estimator in estimators_ext:
                    try:
                        C_est, info = estimator(X_sub)
                        print(f"{info['estimator']}, params: {info['params']}")
                        comp = compare_cov(C_true, C_est)
                        print(f"mse: {comp['mse']}, PDF: {comp['pdf']}")
                        sim_results = []
                        sim_results.append({
                            "Process": proc_name,
                            "n": sample_size,
                            "p": proc_len,
                            "draw": draw_idx,
                            "iter": iter,
                            **info,
                            **comp
                        })
                        save_statistics(f"sim-{iter}", timestamp, results_folder, sim_results)
                    except: 
                        print("Could not make estimator")
                    iter = iter+1

# %% Experiment with real data and 10 fold split
from datetime import datetime
import numpy as np
from sklearn.model_selection import StratifiedKFold

generate_data = False
results_folder = "./results"
dataset_folder = "./datasets/UCRArchive_2018"
timestamp = datetime.today().strftime('%Y%m%d_%H%M%S%f')[:-3]

dataset_names = [
    "Wafer", # 152 Features
    "ECG200", # 96 Features
    "Yoga", # 426 Features
    "Strawberry", # 235 Features
    "FacesUCR", # 131 Features
    "Plane", # 144 Features
    #"Fungi",
    #"StarLightCurves",
    #"EthanolLevel",
    #"FreezerRegularTrain",
    #"PowerCons",
    #"SmallKitchenAppliances",
    #"ScreenType",
    #"MoteStrain",
    #"Computers",
    #"ShapesAll"
]

epsilon_values = [0.01, 0.05, 0.1, 0.15, 0.25]

classifiers = [
    MatrixClassifier("Euclidean", make_identity_cov),
    MatrixClassifier("Maha. diag.", make_sample_diag_cov),
    MatrixClassifier("Maha. sample", make_sample_cov),
    MatrixClassifier("Maha. SB", make_sample_band_cov),
    MatrixClassifier("Maha. LW", make_lw_cov),
    MatrixClassifier("Maha. CB", make_chol_band_cov),
    *[ (lambda epsilon: StateSpaceClassifier("Maha. CB + SS", lambda X: make_ss_approx(make_chol_band_cov, X, epsilon)))(epsilon) for epsilon in epsilon_values],
    *[ (lambda epsilon: StateSpaceClassifier("Maha. sample + SS", lambda X: make_ss_approx(make_sample_cov, X, epsilon)))(epsilon) for epsilon in epsilon_values],
    *[ (lambda epsilon: StateSpaceClassifier("Maha. LW + SS", lambda X: make_ss_approx(make_lw_cov, X, epsilon)))(epsilon) for epsilon in epsilon_values]
]

iter = 0
real_results = []
for dataset_name in (dataset_names if generate_data else []):
    dataset = load_UCR2018(dataset_folder, dataset_name)
    x = np.vstack([ # We use the complete dataset and make our own splits
        dataset["TRAIN"]["X"], 
        dataset["TEST"]["X"]
    ])
    y = np.hstack([
        dataset["TRAIN"]["Y"], 
        dataset["TEST"]["Y"]
    ])
    for classifier in classifiers:
        print(f"Clf: {classifier.name}")
        skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
        split_idx = 0
        for train_index, test_index in skf.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            try:
                classifier.fit(x_train, y_train)
                y_pred = classifier.predict(x_test)
                cm = confusion_matrix(y_pred, y_test, classifier.labels)
                print(f"Acc: {accuracy(cm)}")

                real_results.append({
                    "split": split_idx,
                    "dataset": dataset_name,
                    "p": x.shape[1],
                    f"n_\\mathrm{{train}}": x_train.shape[0],
                    f"n_\\mathrm{{test}}": x_test.shape[0], 
                    "classifier": classifier.name,
                    "acc": accuracy(cm),
                    "precision": precision(cm),
                    "recall": recall(cm),
                    **classifier.info
                })
            except:
                print("Could not fit classifier")
            split_idx = split_idx + 1
        
        save_statistics(f"real_kfold-{iter}_{classifier.name}_{dataset_name}", timestamp, results_folder, real_results)
        iter = iter + 1
        real_results = []

# Experiment with split proposed by dataset

generate_data = False

classifiers = [
    MatrixClassifier("Euclidean", make_identity_cov),
    MatrixClassifier("Maha. diag.", make_sample_diag_cov),
    MatrixClassifier("Maha. sample", make_sample_cov),
    MatrixClassifier("Maha. SB", make_sample_band_cov),
    MatrixClassifier("Maha. LW", make_lw_cov),
    MatrixClassifier("Maha. CB", make_chol_band_cov),
    *[ (lambda epsilon: StateSpaceClassifier("Maha. CB + SS", lambda X: make_ss_approx(make_chol_band_cov, X, epsilon)))(epsilon) for epsilon in epsilon_values],
    *[ (lambda epsilon: StateSpaceClassifier("Maha. sample + SS", lambda X: make_ss_approx(make_sample_cov, X, epsilon)))(epsilon) for epsilon in epsilon_values],
    *[ (lambda epsilon: StateSpaceClassifier("Maha. LW + SS", lambda X: make_ss_approx(make_lw_cov, X, epsilon)))(epsilon) for epsilon in epsilon_values]
]

iter = 0
real_results = []
for dataset_name in (dataset_names if generate_data else []):
    dataset = load_UCR2018(dataset_folder, dataset_name)
    x_train = dataset["TRAIN"]["X"]
    x_test = dataset["TEST"]["X"]
    y_train = dataset["TRAIN"]["Y"]
    y_test = dataset["TEST"]["Y"]
    for classifier in classifiers:
        print(f"Clf: {classifier.name}")
        try:
            classifier.fit(x_train, y_train)
            y_pred = classifier.predict(x_test)
            cm = confusion_matrix(y_pred, y_test, classifier.labels)
            print(f"Acc: {accuracy(cm)}")

            real_results.append({
                "dataset": dataset_name,
                "p": x_train.shape[1],
                f"n_\\mathrm{{train}}": x_train.shape[0],
                f"n_\\mathrm{{test}}": x_test.shape[0], 
                "classifier": classifier.name,
                "acc": accuracy(cm),
                "precision": precision(cm),
                "recall": recall(cm),
                **classifier.info
            })
        except Exception as e:
            print("Could not fit classifier")
            print(str(e))
        
        save_statistics(f"real_orig-{iter}_{classifier.name}_{dataset_name}", timestamp, results_folder, real_results)
        iter = iter + 1
        real_results = []

# %% Visualize / Analyze real experiment data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

for split in ["kfold", "orig"]:
    df_statistics = load_statistics(f"real_{split}-*", results_folder)
    datasets = df_statistics["dataset"].unique()

    for dataset in datasets:
        print(dataset)
        df = df_statistics[df_statistics["dataset"] == dataset].copy()
        n = int(df[f"n_\\mathrm{{train}}"].mean())
        p = int(df["p"].mean())
        df.loc[:, f"\\frac{{\\bar{{d}}^2}}{{\\mathrm{{bw}}}}"] = \
            np.array(df[f"\\bar{{d}}"])**2 / df["bw"]
        df = df.groupby(["estimator", "\\epsilon"], dropna=False).mean()
        df = df[["acc", "precision", "recall", "params", "bw", f"\\bar{{d}}", f"\\frac{{\\bar{{d}}^2}}{{\\mathrm{{bw}}}}"]]
        print(df.head())
        df.to_excel(f"{results_folder}/table-{split}-{dataset}-p{p}-n{n}.xlsx")

# %% Visualize simulation experiment data
df_statistics = load_statistics(f"sim-*", results_folder)
# %%
n_values = df_statistics["n"].unique()
p_values = df_statistics["p"].unique()

pal = [
    (0.7, 0.2, 0.1, 1.0), 
    (0.9, 0.5, 0.3, 0.5), 
    (0.0, 0.5, 0.0, 1.0),
    (0.2, 0.9, 0.2, 0.5), 
    (0.2, 0.1, 0.7, 1.0), 
    (0.5, 0.3, 0.9, 0.5),
]

for n in n_values: 
    for p in p_values:
        df = df_statistics[
            (df_statistics["n"] == n) &
            (df_statistics["p"] == p) ].copy()
        df.loc[:,f"log$_{{10}}$(params)"] = np.log10(df["params"])
        df.loc[:,f"log$_{{10}}$(mse)"] = np.log10(df["mse"])
        df = df.groupby(["estimator", "\\epsilon", "Process"], dropna=False).mean()
        df = df.reset_index()
        df["\\epsilon"] = df["\\epsilon"].fillna("na")
        df.loc[:, "is approx."] = df.apply(lambda row: row["\\epsilon"] != "na", axis=1)
        df =  df[df["estimator"].isin([
                "Ledoit Wolf + SS", 
                "Cholesky band + SS",
                "Ledoit Wolf",
                "Cholesky band",
                #"Sample", 
                #"Sample + SS" 
                ])]
        n_cols = len(df["estimator"].unique())
        g = sns.relplot(
            data=df, 
            x=f"log$_{{10}}$(params)", y=f"log$_{{10}}$(mse)", 
            hue="estimator", size="\\epsilon",
            style="is approx.", col="Process", 
            sizes=(100,20), palette=pal[0:n_cols],
            facet_kws={'sharey': False})
        g.fig.suptitle(f"p={int(df['p'].mean())}, n={int(df['n'].mean())}", y=1.1, x=0.45)
        
            
# %%
