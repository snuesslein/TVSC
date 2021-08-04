# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import OAS
from tvsclib.strict_system import StrictSystem
from tvsclib.toeplitz_operator import ToeplitzOperator
from tvsclib.system_identification_svd import SystemIdentificationSVD

# %%
class DistanceBasedClasifier:
    def __init__(self, base_dir:str, dataset_name:str, no_split:bool=False):
        self.base_dir = base_dir
        self.dataset_name = dataset_name
        self.x_train, self.y_train = self.loadDataset(
            f"{base_dir}/{dataset_name}/{dataset_name}_TRAIN.tsv")
        self.x_test, self.y_test = self.loadDataset(
            f"{base_dir}/{dataset_name}/{dataset_name}_TEST.tsv")
        if no_split:
            self.x_train = np.hstack([self.x_train, self.x_test])
            self.y_train = np.hstack([self.y_train, self.y_test])
    
    def calcClassMean(self, label):
        return self.x_train[:,self.y_train == label].mean(axis=1).reshape((-1,1))

    def calcClassCovariances(self):
        y = self.y_train
        x = self.x_train
        class_covariance = {}
        for class_label in np.unique(y):
            class_data = x[:, y == class_label].transpose()
            #print(f"Samples for class {class_label}: {class_data.shape[0]}")
            S = OAS().fit(class_data).covariance_
            class_covariance[class_label] = S
        return class_covariance

    def loadDataset(self, filename:str):
        data = np.loadtxt(filename)
        y = data[:,0].transpose()
        x = data[:,1:].transpose()
        return (x,y)
    
    def calcDistances(self, x_in):
        raise NotImplementedError()
    
    def fit(self):
        raise NotImplementedError()
    
    def predictClasses(self, x_in):
        distances,labels = self.calcDistances(x_in)
        return labels[np.argmin(distances,axis=0)]
    
    def calcAccuracy(self, include_train:bool=False):
        if include_train:
            x_hat = np.hstack([self.x_test, self.x_train])
            y = np.hstack([self.y_test, self.y_train])
        else:
            x_hat = self.x_test
            y = self.y_test
        y_hat = self.predictClasses(x_hat)
        n_pos = np.sum(y_hat.flatten() == y)
        return n_pos / y.shape[0]

    def calcConfusionMatrix(self, include_train:bool=False):
        if include_train:
            x_hat = np.hstack([self.x_test, self.x_train])
            y = np.hstack([self.y_test, self.y_train])
        else:
            x_hat = self.x_test
            y = self.y_test
        y_hat = self.predictClasses(x_hat)
        labels = np.unique(y)
        confusion = np.zeros((labels.shape[0], labels.shape[0]))
        i = 0
        for label_pred in labels:
            #print(f"Total {label_pred}: {np.sum(y == label_pred)}")
            j = 0
            for label_true in labels:
                confusion[i,j] = np.sum(
                    (y_hat.flatten() == label_pred).astype("int32") \
                        * (y == label_true))
                j = j+1
            i = i+1
        return (confusion.astype("int32"), labels)
    
    def getLabels(self):
        return np.unique(np.hstack([self.y_test, self.y_train]))
    
    def getNFeatures(self):
        return self.x_train.shape[0]


# %%
class MahalanobisClassifier(DistanceBasedClasifier):
    def __init__(self, base_dir:str, dataset_name:str, no_split:bool=False):
        super().__init__(base_dir, dataset_name, no_split)
        self.inv_covariances = {}
    
    def fit(self):
        for label,covariance in self.calcClassCovariances().items():
            self.inv_covariances[label] = np.linalg.inv(covariance)

    def calcRequirements(self):
        flops = 0
        n_params = 0
        for _,inv_covariance in self.inv_covariances.items():
            flops = flops \
                + 2*inv_covariance.shape[0] - 1 \
                + inv_covariance.shape[0]**2 + inv_covariance.shape[0] - 1 # Lower triangular
            n_params = n_params \
                + inv_covariance.shape[0]*(inv_covariance.shape[0] + 1)/2
        return (flops, n_params)

    def calcDistances(self, x_in):
        test_distances = []
        test_labels = []
        for label,inv_covariance in self.inv_covariances.items():
            x_in_0 = (x_in - self.calcClassMean(label))
            distances = x_in_0.transpose() @ np.linalg.cholesky(inv_covariance)
            test_labels = [*test_labels, label]
            test_distances = [*test_distances, np.linalg.norm(distances, axis=1)]
        test_distances = np.vstack(test_distances)
        test_labels = np.vstack(test_labels)
        return (test_distances, test_labels)


# %%
class MahalanobisSSClassifier(DistanceBasedClasifier):
    def __init__(self, base_dir:str, dataset_name:str, 
        no_split:bool=False, max_states_local:int=5, epsilon:float=0.05,
        block_size:int=2):
        super().__init__(base_dir, dataset_name, no_split)
        self.max_states_local = max_states_local
        self.block_size = block_size
        self.epsilon = epsilon
        self.inv_systems = {}

    def fit(self):
        for label,covariance in self.calcClassCovariances().items():
            C_sq = np.linalg.cholesky(covariance)
            io_dims = [ 
                *([self.block_size]*int(C_sq.shape[0]/self.block_size)), 
                *([1]*int(C_sq.shape[0]%self.block_size)) ]
            T = ToeplitzOperator(
                C_sq, 
                io_dims,
                io_dims)
            S = SystemIdentificationSVD(
                T, max_states_local=self.max_states_local,
                relative=True, epsilon=self.epsilon)
            system = StrictSystem(
                system_identification=S, 
                causal=True)
            To,_ = system.outer_inner_factorization()
            self.inv_systems[label] = To.arrow_reversal()

    def calcRequirements(self):
        flops = 0
        n_params = 0
        for _,inv_system in self.inv_systems.items():
            flops = flops + 2*np.sum(inv_system.dims_out) - 1 # Calculate squared norm
            for stage in inv_system.stages:
                flops = flops \
                    + 2*stage.A_matrix.shape[0]*stage.A_matrix.shape[1] - 1 \
                    + 2*stage.B_matrix.shape[0]*stage.B_matrix.shape[1] - 1 \
                    + 2*stage.C_matrix.shape[0]*stage.C_matrix.shape[1] - 1 \
                    + 2*stage.D_matrix.shape[0]*stage.D_matrix.shape[1] - 1 \
                    + (stage.A_matrix.shape[0] + stage.B_matrix.shape[0]) # Addition op's in x' = Ax + Bu and y = Cx + Du
                n_params = n_params \
                    + stage.A_matrix.shape[0]*stage.A_matrix.shape[1] \
                    + stage.B_matrix.shape[0]*stage.B_matrix.shape[1] \
                    + stage.C_matrix.shape[0]*stage.C_matrix.shape[1] \
                    + stage.D_matrix.shape[0]*stage.D_matrix.shape[1]
        return (flops, n_params)

    def calcDistances(self, x_in):
        test_distances = []
        test_labels = []
        for label,inv_system in self.inv_systems.items():
            x_in_0 = (x_in - self.calcClassMean(label))
            distances = np.zeros(x_in_0.shape[1])
            for col_idx in range(x_in_0.shape[1]):
                _,y_system = inv_system.compute(x_in_0[:,col_idx].reshape((-1,1)))
                distances[col_idx] = np.linalg.norm(y_system)
            test_labels = [*test_labels, label]
            test_distances = [*test_distances, distances]
        test_distances = np.vstack(test_distances)
        test_labels = np.vstack(test_labels)
        return (test_distances, test_labels)


# %%
class EuclideanClassifier(DistanceBasedClasifier):
    def __init__(self, base_dir:str, dataset_name:str, no_split:bool=False):
        super().__init__(base_dir, dataset_name, no_split)

    def fit(self):
        pass

    def calcRequirements(self):
        n = self.calcClassCovariances().values()[0].shape[0]
        flops = 2*n - 1
        n_params = 0
        return (flops, n_params)

    def calcDistances(self, x_in):
        test_distances = []
        test_labels = []
        for label in self.getLabels():
            d = (x_in - self.calcClassMean(label))
            distances = np.diag(d.transpose() @ d)
            test_labels = [*test_labels, label]
            test_distances = [*test_distances, distances]
        test_distances = np.vstack(test_distances)
        test_labels = np.vstack(test_labels)
        return (test_distances, test_labels)


# %%
current_time = datetime.utcnow().strftime('%Y-%m-%d_%H%M%S%f')[:-3]
dataset_folder = ".local/UCRArchive_2018"
subfolders = [f.path for f in os.scandir(dataset_folder) if f.is_dir()]
skip_datasets = [
    'Adiac',
    'Missing_value_and_variable_length_datasets_adjusted']
for folder in subfolders:
    dataset_name = os.path.basename(folder)

    #dataset_name = "GunPoint"
    print(f"Dataset: {dataset_name}")
    
    if skip_datasets.count(dataset_name) != 0:
        print(f"Skip dataset becuase it is in skipping list")
        continue

    euclidian = EuclideanClassifier(dataset_folder, dataset_name, no_split=True)
    if euclidian.getNFeatures() > 500:
        print("Skip dataset because too many features")
        continue
    euclidian.fit()
    acc_base = euclidian.calcAccuracy(include_train=True)
    print(f"Acc euclidian: {acc_base}")

    confusion_eucl,labels = euclidian.calcConfusionMatrix(include_train=True)
    print("Confusion euclidian:")
    print(confusion_eucl.round())
    #fig, ax = plt.subplots(figsize=(7.5, 7.5))
    #ax.matshow(confusion_eucl, cmap=plt.cm.Blues, alpha=0.3)
    #for i in range(confusion_eucl.shape[0]):
    #    for j in range(confusion_eucl.shape[1]):
    #        ax.text(x=j, y=i,s=confusion_eucl[i, j], va='center', ha='center', size='xx-large')
    #ax.set_xticklabels(['']+list(labels))
    #ax.set_yticklabels(['']+list(labels))
    #plt.show()

    mahalanobis = MahalanobisClassifier(dataset_folder, dataset_name, no_split=True)
    mahalanobis.fit()
    #for label,covariance in mahalanobis.calcClassCovariances().items():
    #    plt.imshow(covariance, cmap='hot', interpolation='nearest')
    #    plt.title(f"Label {label}, C")
    #    plt.show()
    #    plt.imshow(np.linalg.inv(covariance), cmap='hot', interpolation='nearest')
    #    plt.title(f"Label {label}, inv(C)")
    #    plt.show()
    acc = mahalanobis.calcAccuracy(include_train=True)
    print(f"Acc mahalanobis: {acc}")
    if acc < 0.6:
        print("Skip dataset because distance based classification doenst work")
        continue

    confusion_maha,labels = mahalanobis.calcConfusionMatrix(include_train=True)
    print("Confusion mahalanobis:")
    print(confusion_maha.round())
    #fig, ax = plt.subplots(figsize=(7.5, 7.5))
    #ax.matshow(confusion_maha, cmap=plt.cm.Blues, alpha=0.3)
    #for i in range(confusion_maha.shape[0]):
    #    for j in range(confusion_maha.shape[1]):
    #        ax.text(x=j, y=i,s=confusion_maha[i, j], va='center', ha='center', size='xx-large')
    #ax.set_xticklabels(['']+list(labels))
    #ax.set_yticklabels(['']+list(labels))
    #plt.show()


    for block_size in [8,4,2]:
        print(f"block size: {block_size}")
        # Determine max state dimension of initial system
        max_states = np.ceil(mahalanobis.getNFeatures() / 10)
        # Initialize Grid search
        #block_size = 8
        epsilon_stepsize = 0.025
        epsilon_steps = 7
        epsilon_cols = []
        acc_cols = []
        confusion_cols = []
        speedup_cols = []
        memory_cols = []
        accloss_cols = []
        d_max = max_states.astype("int32")
        while d_max > 0:
            keep_reducing = True
            epsilon_col = []
            acc_col = []
            confusion_col = []
            speedup_col = []
            memory_col = []
            accloss_col = []
            for epsilon_step in range(epsilon_steps,-1,-1):
                epsilon = epsilon_stepsize*epsilon_step
                print(f"d_max: {d_max}, epsilon: {epsilon}")
                mahalanobisSS = MahalanobisSSClassifier(
                    dataset_folder, dataset_name, 
                    no_split=True, max_states_local=d_max, epsilon=epsilon,
                    block_size=block_size)
                mahalanobisSS.fit()
                # Determine max state dimension of current system
                state_dim_max = 0
                for system in mahalanobisSS.inv_systems.values():
                    state_dim_max = max(state_dim_max, np.max(system.dims_state))
                # Check if epsilon got too big
                if state_dim_max < d_max:
                    print(f"Skip epsilon: {epsilon}")
                    continue
                
                epsilon_col = [*epsilon_col, epsilon]

                acc_ss = mahalanobisSS.calcAccuracy(include_train=True)
                acc_col = [*acc_col, acc_ss]
                print(f"Acc: {acc_col[-1]}")

                confusion,labels = mahalanobisSS.calcConfusionMatrix(include_train=True)
                confusion_col = [*confusion_col, confusion]
                #fig, ax = plt.subplots(figsize=(7.5, 7.5))
                #ax.matshow(confusion, cmap=plt.cm.Blues, alpha=0.3)
                #for i in range(confusion.shape[0]):
                #    for j in range(confusion.shape[1]):
                #        ax.text(x=j, y=i,s=confusion[i, j], va='center', ha='center', size='xx-large')
                #ax.set_xticklabels(['']+list(labels))
                #ax.set_yticklabels(['']+list(labels))
                #plt.show()

                flops_direct,mem_direct = mahalanobis.calcRequirements()
                flops_ss,mem_ss = mahalanobisSS.calcRequirements()
                speedup_col = [*speedup_col, flops_direct/flops_ss]
                memory_col = [*memory_col, mem_direct/mem_ss]
                accloss_col = [*accloss_col, acc-acc_ss]
                print(f"Theoretical speedup: {speedup_col[-1]}")
                print(f"Memory reduction: {memory_col[-1]}")
                print(f"Accuracy loss: {accloss_col[-1]}")
                
                if speedup_col[-1] < 1:
                    print(f"Break performance: {speedup_col[-1]}")
                    break

            epsilon_cols = [*epsilon_cols, np.array(epsilon_col).reshape((-1,1))]
            acc_cols = [*acc_cols, np.array(acc_col).reshape((-1,1))]
            confusion_cols = [*confusion_cols, confusion_col]
            speedup_cols = [*speedup_cols, np.array(speedup_col).reshape((-1,1))]
            memory_cols = [*memory_cols, np.array(memory_col).reshape((-1,1))]
            accloss_cols = [*accloss_cols, np.array(accloss_col).reshape((-1,1))]
            
            d_max = d_max - 1
                
        # Save stats
        statistics_dir = './results'
        statistics_filename = \
            f"{statistics_dir}/{dataset_name}_{block_size}_{current_time}.npy"
        statistic_data = {
            'labels': labels,
            'confusion_eucl': confusion_eucl,
            'confusion_maha': confusion_maha,
            'acc_base': acc_base,
            'acc': acc,
            'dataset_name': dataset_name,
            'max_states': max_states,
            'block_size': block_size,
            'epsilon_stepsize': epsilon_stepsize,
            'epsilon_steps': epsilon_steps,
            'epsilon_cols': epsilon_cols,
            'acc_cols': acc_cols,
            'confusion_cols': confusion_cols,
            'speedup_cols': speedup_cols,
            'memory_cols': memory_cols,
            'accloss_cols': accloss_cols
        }

        np.save(statistics_filename, statistic_data)
# %%
data_loaded = np.load(statistics_filename, allow_pickle=True).item()
