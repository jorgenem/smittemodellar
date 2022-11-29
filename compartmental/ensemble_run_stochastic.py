import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from stochastic import *


# Run many realisations of the stochastic compartmental model



# Set parameters and initial conditions
T = 30
dt = 0.1
beta = 1.8
gamma = 0.8
N = 1000
I0 = 50
R0 = 0

# Instantiate model
sir = SIR(T=T, dt=dt, beta=beta, gamma=gamma, N=N, I0=I0, R0=R0)


# Run Nsim times and store result in dict which will be converted to pandas dataframe
Nsim = 20
results_list = []
tstart = time.perf_counter()
for i in range(Nsim):
    t, S, I, R = sir.run_model()
    results_list.append(
        pd.DataFrame({
            't': t,
            'S': S,
            'I': I,
            'R': R,
            'sim': i
        })
    )
tend = time.perf_counter()
print("Ran {:d} stochastic realisations in {:.3f} seconds.".format(Nsim, tend-tstart))
results = pd.concat(results_list)

# Pivot results long for plotting
results_long = pd.melt(results, id_vars = ['t', 'sim'], var_name='compartment', value_name='count')
results_long['grouping'] = results_long['compartment'].astype(str) + "_" + results_long['sim'].astype(str)

# Plot using plotnine (ggplot2)
from plotnine import *
(ggplot(results_long)
+ aes(x='t', y='count', colour='compartment', group='grouping')
+ geom_line(alpha=0.7)
)