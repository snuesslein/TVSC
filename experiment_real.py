# %% Experiment with real data and 10 fold split
from datetime import datetime
import numpy as np
from sklearn.model_selection import StratifiedKFold
from experiment.estimators import *
from experiment.classifiers import *
from experiment.datahandling import *

generate_data = False
results_folder = "./results"
dataset_folder = "./datasets/UCRArchive_2018"
timestamp = datetime.today().strftime('%Y%m%d_%H%M%S%f')[:-3]

dataset_names = [
    "ECG200", 
    "Wafer", 
    "Strawberry",
]

epsilon_values = [
    0.05, 
    0.1, 
    0.15,
    0.3, 
    0.6
]

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
import seaborn as sns
import matplotlib.pyplot as plt

for split in ["kfold", "orig"]:
    df_statistics = load_statistics(f"real_{split}-*", results_folder)

    if df_statistics is None:
        continue

    datasets = df_statistics["dataset"].unique()

    for dataset in datasets:
        print(dataset)
        df = df_statistics[df_statistics["dataset"] == dataset].copy()
        n = int(df[f"n_\\mathrm{{train}}"].mean())
        p = int(df["p"].mean())
        df = df.groupby(["estimator", "\\epsilon"], dropna=False).mean()
        df = df[["acc", "precision", "recall", "params", "flops", "bw", f"\\bar{{d}}", "speedup", "savings"]]
        print(df.head())
        df.to_excel(f"{results_folder}/table-real-{split}-{dataset}-p{p}-n{n}.xlsx")
# %%
