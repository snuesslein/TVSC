# %% Experiment with simulated data
from datetime import datetime
import numpy as np
from experiment.processes import *
from experiment.estimators import *
from experiment.compare import *
from experiment.datahandling import *

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
    0.2,
#    0.3,
    0.4,
#    0.5,
    0.6,
#    0.7
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
no_draws = 30
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



# %% Visualize simulation experiment data
df_statistics = load_statistics(f"sim-*", results_folder)
# %%
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

n_values = df_statistics["n"].unique()
p_values = df_statistics["p"].unique()
proc_values = df_statistics["Process"].unique()

pal = [
    (0.7, 0.2, 0.1, 1.0), 
    (0.0, 0.5, 0.0, 1.0),
    (0.2, 0.1, 0.7, 1.0), 
]

for n in n_values: 
    for p in p_values:
        for proc in proc_values:
            df = df_statistics[
                (df_statistics["n"] == n) &
                (df_statistics["p"] == p) &
                (df_statistics["Process"] == proc) ].copy()
            df.loc[:,f"log$_{{10}}$(params)"] = np.log10(df["params"])
            df.loc[:,f"log$_{{10}}$(mse)"] = np.log10(df["mse"])
            df = df.groupby(["estimator", "\\epsilon"], dropna=False).mean()
            df = df.reset_index()
            df.loc[:,"$\\epsilon$"] = df["\\epsilon"]
            df_ss =  df[df["estimator"].isin([
                    "Ledoit Wolf + SS", 
                    "Cholesky band + SS",
                    "Sample + SS" 
                    ])]
            n_colors = len(df_ss["estimator"].unique())
            df_mat =  df[df["estimator"].isin([
                    "Ledoit Wolf", 
                    "Cholesky band",
                    "Sample", 
                    ])]
            df_mat.loc[:, "estimator"] = df_mat.apply(lambda row: 
                row["estimator"] \
                + (f" ($\\bar{{bw}}={df_mat['bw'].mean():.2f}$)" if row["estimator"] == "Cholesky band" else ""), 
                axis=1)
            fig = plt.figure()
            gs0 = matplotlib.gridspec.GridSpec(2,1,figure=fig)
            ax1 = fig.add_subplot(gs0[0,:])
            ax2 = fig.add_subplot(gs0[1,:])
            sns.lineplot(ax=ax1, data=df_ss, x="$\\epsilon$", y=f"log$_{{10}}$(mse)", 
                hue="estimator", palette=pal[0:n_colors], legend=False)
            pal_idx = 0
            for estimator in df_mat["estimator"].unique():
                y = df_mat[df_mat["estimator"] == estimator][f"log$_{{10}}$(mse)"].mean()
                ax1.axhline(y, ls="--", c=pal[pal_idx])
                pal_idx = pal_idx + 1
                if pal_idx >= n_colors:
                    break
            sns.lineplot(ax=ax2, data=df_ss, x="$\\epsilon$", y=f"log$_{{10}}$(params)", 
                hue="estimator", palette=pal[0:n_colors], legend=False)
            pal_idx = 0
            handles,labels = ax2.get_legend_handles_labels()
            for estimator in df_mat["estimator"].unique():
                y = df_mat[df_mat["estimator"] == estimator][f"log$_{{10}}$(params)"].mean()
                ax2.axhline(y, ls="--", c=pal[pal_idx])
                patch = matplotlib.patches.Patch(color=pal[pal_idx], label=estimator)
                handles.append(patch)
                pal_idx = pal_idx + 1
                if pal_idx >= n_colors:
                    break
            line1 = matplotlib.lines.Line2D([0],[0], label="w.o. approx.", color="k", 
                linestyle="--")
            line2 = matplotlib.lines.Line2D([0],[0], label="w. approx.", color="k", 
                linestyle="-")
            handles.extend([line1, line2])
            plt.legend(handles=handles,
                loc='upper center', bbox_to_anchor=(0.5, -0.35),
                fancybox=True, shadow=True, ncol=2)
            plt.suptitle(f"{proc}, $p={p}$, $n={n}$")
            plt.show()
# %%
