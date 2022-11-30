import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed

from agent_based import *


def wrapper_call(i):
    # Wrapper to call the ABM for parallel multiprocessing loop
    # Instantiate model:
    abm = ABM(
        N_pop=1000,
        gamma_numcontacts_k=4,
        gamma_numcontacts_scale=2,
        gamma_recoverytime_k=2,
        gamma_recoverytime_scale=1,
        N_initial_infected=1,
        dt=0.1
    )
    # Generate population of susceptibles and seed initial infections:
    abm.reset_run() 
    # Run infectious disease model forward in time:
    t, incidence, I, S, R  = abm.run_abm(
        beta = 1.4,
        Tmax = 10
        )
    dt_results = {
        't': t,
        'S': S,
        'I': I,
        'R': R,
        'incidence': incidence,
        'sim': i*np.ones(len(t))
    }
    return dt_results


N_runs = 2
# Two options for main execution:
# 1. Serial run single CPU:
# list_results = []
# for i in range(N_runs):
#     list_results.append(wrapper_call(i))
#  2. Or rather parallel?
list_results = Parallel(n_jobs=2)(delayed(wrapper_call)(i) for i in range(N_runs))


# Make one dataframe with all results
# df_results = [pd.DataFrame(dt) for key, dt in dt_results.items()]
df_results = [pd.DataFrame(dt) for dt in list_results]
df_results = pd.concat(df_results)

# # Plot
# # Incidence
# (pln.ggplot(df_results)
# + pln.aes(x='timearray', y = 'incidence', group = 'sim', colour = 'sim')
# + pln.geom_line()
# )
# # Prevalence
# (pln.ggplot(df_results)
# + pln.aes(x='t', y = 'prevalence', group = 'sim', colour = 'sim')
# + pln.geom_line()
# )


# Pivot results long for plotting
results_long = pd.melt(df_results.loc[:, df_results.columns!="incidence"], id_vars = ['t', 'sim'], var_name='compartment', value_name='count')
results_long['grouping'] = results_long['compartment'].astype(str) + "_" + results_long['sim'].astype(str)

# Plot using plotnine (ggplot2)
from plotnine import *
(ggplot(results_long)
+ aes(x='t', y='count', colour='compartment', group='grouping')
+ geom_line(alpha=0.7)
)