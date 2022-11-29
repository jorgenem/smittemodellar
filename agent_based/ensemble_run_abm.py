import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotnine as pln

from agent_based import *


N_runs = 20
dt_results = {}
for i in range(N_runs):
    # Instantiate model:
    abm = ABM(
        N_pop=1000,
        gamma_numcontacts_k=10,
        gamma_numcontacts_scale=0.2,
        gamma_recoverytime_k=3,
        gamma_recoverytime_scale=1,
        N_initial_infected=1,
        dt=0.1
    )
    # Generate population of susceptibles and seed initial infections:
    abm.reset_run() 
    # Run infectious disease model forward in time:
    timearray, prevalence, incidence = abm.run_abm(
        beta = 1,
        Tmax = 10
        )


    dt_results[i] = {
        'timearray': timearray,
        'prevalence': prevalence,
        'incidence': incidence,
        'sim': i*np.ones(len(timearray))
    }



# Make one dataframe with all results
df_results = [pd.DataFrame(dt) for key, dt in dt_results.items()]
df_results = pd.concat(df_results)

# Plot
# Incidence
(pln.ggplot(df_results)
+ pln.aes(x='timearray', y = 'incidence', group = 'sim', colour = 'sim')
+ pln.geom_line()
)
# Prevalence
(pln.ggplot(df_results)
+ pln.aes(x='timearray', y = 'prevalence', group = 'sim', colour = 'sim')
+ pln.geom_line()
)