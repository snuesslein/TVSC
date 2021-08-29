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