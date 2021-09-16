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
    100
]
sample_sizes = [50, 100, 150, 200]
epsilon_values = [
    0.05,
    0.1,
    0.3,
    0.6,
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
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
import seaborn as sns

n_values = df_statistics["n"].unique()
p_values = df_statistics["p"].unique()

for n in n_values: 
    for p in p_values:
        df = df_statistics[
            (df_statistics["n"] == n) &
            (df_statistics["p"] == p) &
            df_statistics["estimator"].isin([
                "Cholesky band + SS",
                "Ledoit Wolf + SS",
                "Cholesky band",
                "Sample + SS",
                "Sample",
                "Ledoit Wolf"
            ]) ].copy()
        
        for proc in np.sort(df["Process"].unique()):
            df_table = df[df["Process"] == proc]
            df_table = df_table.groupby(["estimator", "\\epsilon"], dropna=False).median()
            df_table = df_table[["mse", "params", "flops", "bw", f"\\bar{{d}}", "speedup", "savings"]]
            print(df_table.head())
            df_table.to_excel(f"{results_folder}/table-sim-{proc}-p{p}-n{n}.xlsx")
            
        df = df[~df["estimator"].isin(["Sample + SS", "Sample"])]
        df.loc[:,"$\\epsilon$"] = df["\\epsilon"]
        df.loc[:,f"log$_{{10}}$(mse)"] = np.log10(df["mse"])
        df.loc[:,f"log$_{{10}}$(params)"] = np.log10(df["params"])
        df.loc[:,f"log$_{{10}}$(flops)"] = np.log10(df["flops"])

        fig,ax = plt.subplots(3,2, sharex=True, sharey=False)
        col_idx = 0
        for proc in np.sort(df["Process"].unique()):
            df2 = df[df["Process"] == proc]
            df3 = df2[~df2["estimator"].isin(["Ledoit Wolf", "Cholesky band"])]
            df4 = df2[df2["estimator"].isin(["Ledoit Wolf", "Cholesky band"])]
            ax[0,col_idx].set_title(proc)
            sns.lineplot(data=df3, x="$\\epsilon$", y=f"log$_{{10}}$(params)", hue="estimator", ax=ax[0,col_idx])
            ax[0,col_idx].axhline(df4[df4["estimator"] == "Cholesky band"][f"log$_{{10}}$(params)"].mean(), color='k', ls="--")
            ax[0,col_idx].axhline(df4[df4["estimator"] == "Ledoit Wolf"][f"log$_{{10}}$(params)"].mean(), color='k', ls="-.")
            sns.lineplot(data=df3, x="$\\epsilon$", y=f"log$_{{10}}$(flops)", hue="estimator", ax=ax[1,col_idx])
            ax[1,col_idx].axhline(df4[df4["estimator"] == "Cholesky band"][f"log$_{{10}}$(flops)"].mean(), color='k', ls="--")
            ax[1,col_idx].axhline(df4[df4["estimator"] == "Ledoit Wolf"][f"log$_{{10}}$(flops)"].mean(), color='k', ls="-.")
            sns.lineplot(data=df3, x="$\\epsilon$", y="mse", hue="estimator", ax=ax[2,col_idx])
            ax[2,col_idx].axhline(df4[df4["estimator"] == "Cholesky band"]["mse"].mean(), color='k', ls="--")
            ax[2,col_idx].axhline(df4[df4["estimator"] == "Ledoit Wolf"]["mse"].mean(), color='k', ls="-.")
            col_idx = col_idx + 1

        h = []
        l = []
        for ax_el in ax.flatten():
            ax_el.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            handles, labels = ax_el.get_legend_handles_labels()
            h = [*h, *handles]
            l = [*l, *labels]
            ax_el.get_legend().remove()
        _,ids = np.unique(l, return_index=True)
        h = [h[i] for i in ids]
        l = [l[i] for i in ids]  
        plt.legend(h,l)

        handles,_ = plt.gca().get_legend_handles_labels()
        handles.extend([Line2D([0],[0],label="Ledoit Wolf", color="k", ls="-.")])
        handles.extend([Line2D([0],[0],label="Cholesky band", color="k", ls="--")])
        plt.legend(handles=handles, loc='upper center', bbox_to_anchor=(-0.35, -0.6), ncol=2)
        plt.suptitle(f"$n={n}$, $p={p}$", y=1.0)
        plt.subplots_adjust(wspace=0.6, hspace = 0.5)
        plt.show()
            
# %%
_,C_ma = processes["MA(4)"](10,200)
_,C_tvarma = processes["tvARMA(2,2)"](10,200)

# using tuple unpacking for multiple Axes
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(C_tvarma, cmap='hot', interpolation='nearest')
ax1.title.set_text("tvARMA(2,2)")
ax2.imshow(C_ma, cmap='hot', interpolation='nearest')
ax2.title.set_text("MA(4)")
plt.show()
# %%
