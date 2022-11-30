import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time


"""
A simple agent-based SIR model
By Jørgen Eriksson Midtbø, Folkehelseinstituttet, github.com/jorgenem
"""



class ABM:
    def __init__(self, N_pop, gamma_numcontacts_k, gamma_numcontacts_scale, gamma_recoverytime_k, gamma_recoverytime_scale, N_initial_infected, dt = 1.0):
        self.N_pop = N_pop
        self.gamma_numcontacts_k = gamma_numcontacts_k
        self.gamma_numcontacts_scale = gamma_numcontacts_scale
        self.gamma_recoverytime_k = gamma_recoverytime_k
        self.gamma_recoverytime_scale = gamma_recoverytime_scale
        self.dt = dt
        self.N_initial_infected = N_initial_infected

        # Instantiate random generator
        self.rng = np.random.default_rng()

    def reset_run(self):
        # Generate population
        self.generate_population()

        # Seed initial infections
        self.seed_infection(N_infected=self.N_initial_infected)

    
    def generate_population(self):
        N_pop = self.N_pop
        self.df_pop = pd.DataFrame({
            "id": np.arange(N_pop), # Unique id of each individual
            "status": np.repeat(np.array(["S"]), self.N_pop), # Epidemiological status
            "timestep_infected": (np.zeros(N_pop)-1).astype(int), # Time point of infection
            "timestep_recovered": (np.zeros(N_pop)-1).astype(int) # Time point of recovery
        })

    def seed_infection(self, N_infected):
        # Select indices of those who should be infected
        infected = self.rng.choice(self.df_pop['id'], size=N_infected, replace=False)
        self.df_pop.loc[infected, 'status'] = "I"
        self.df_pop.loc[infected, 'timestep_infected'] = 0
        self.df_pop.loc[infected, 'timestep_recovered'] = 0 + self.rng.gamma(self.gamma_recoverytime_k, self.gamma_recoverytime_scale, size=N_infected)

    def run_abm(self, beta, Tmax):
        df_pop = self.df_pop # Rename for convenience inside this function, should be just a view into the self.df_pop object which will be modified
        timearray = np.linspace(0, Tmax, int(Tmax/self.dt)) # Make time array

        # Initialise counters
        incidence = np.zeros(len(timearray))
        prevalence_infected = np.zeros(len(timearray))
        prevalence_susceptible = np.zeros(len(timearray))
        prevalence_recovered = np.zeros(len(timearray))

        # Timestep 0 is assumed to be when seed cases happen:
        incidence[0] = np.sum(((df_pop['status'] == "I") & (df_pop['timestep_infected'] == 0)))
        prevalence_infected[0] = np.sum((df_pop['status'] == "I"))
        prevalence_susceptible[0] = np.sum((df_pop['status'] == "S"))
        prevalence_recovered[0] = np.sum((df_pop['status'] == "R"))

        # Then loop from timestep 1 onward:
        # TODO can this get speedup using numba? Would require some rewriting of Pandas dfs to Numpy arrays.
        for it, t in zip(range(1, len(timearray)), timearray[1:]): # it is integer time index, t is clock time, such that t = it*dt. 
            infected = df_pop[df_pop['status'] == "I"]['id']
            susceptible = df_pop[df_pop['status'] == "S"]['id']
            for i in infected: # TODO test parallelising this loop, not caring about double infections of same infectee in multiple chains
                # Draw number of total contact encounters for individual
                N_contacts = int(self.rng.gamma(self.gamma_numcontacts_k, self.gamma_numcontacts_scale, size=1))
                # print("infector i = ", i, ", N_contacts =", N_contacts)
                # Decide how many of these contact encounters resulted in possible transmission
                prob_infectious_contact = 1 - np.exp(-beta*self.dt)
                N_contacts_infected = self.rng.binomial( N_contacts, prob_infectious_contact)
                infectees = self.rng.choice(df_pop['id'], size = N_contacts_infected, replace = False)
                infectees = infectees[df_pop['status'][infectees] == "S"] # Filter out only S individuals among infectees to actually get infected
                df_pop.loc[infectees, 'status'] = "I"
                df_pop.loc[infectees, 'timestep_infected'] = it
                df_pop.loc[infectees, 'timestep_recovered'] = it + (self.rng.gamma(self.gamma_recoverytime_k, self.gamma_recoverytime_scale, size=len(infectees))/self.dt).astype(int) # Deciding recovery time in the future for each individual

            # Who recovers at this timestep?
            recovered = df_pop[df_pop['timestep_recovered'] == it]['id']
            df_pop.loc[recovered, 'status'] = "R"

            # Update counters at this timestep
            incidence[it] = np.sum(((df_pop['status'] == "I") & (df_pop['timestep_infected'] == it)))
            prevalence_infected[it] = np.sum((df_pop['status'] == "I"))
            prevalence_susceptible[it] = np.sum((df_pop['status'] == "S"))
            prevalence_recovered[it] = np.sum((df_pop['status'] == "R"))

            # TODO calculate and store reproduction number?
                


        return timearray, incidence, prevalence_infected, prevalence_susceptible, prevalence_recovered



if __name__ == "__main__":


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
    tstart = time.perf_counter()
    t, incidence, I, S, R = abm.run_abm(
        beta = 0.4,
        Tmax = 20
        )
    tend = time.perf_counter()
    print("Ran agent-based model in {:.3f} seconds.".format(tend-tstart))

    # Plot results
    f, ax = plt.subplots(2)
    ax[0].plot(t, incidence, label='incidence')
    ax[0].set_title('incidence of infections')
    ax[1].plot(t, I, label='I')
    ax[1].plot(t, S, label='S')
    ax[1].plot(t, R, label='R')
    ax[1].set_title('prevalence in compartments')
    ax[1].legend()
    plt.tight_layout()
    plt.show()

    
    # Plot using plotnine (ggplot2)
    dfp = pd.concat([
        pd.DataFrame({"t": t, "count": S, "compartment": "S"}),
        pd.DataFrame({"t": t, "count": I, "compartment": "I"}),
        pd.DataFrame({"t": t, "count": R, "compartment": "R"})
    ])
    from plotnine import *
    (ggplot(dfp)
    + aes(x='t', y='count', colour='compartment')
    + geom_line()
    )
