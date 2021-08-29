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