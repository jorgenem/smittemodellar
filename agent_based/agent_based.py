import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


"""
A simple agent-based SIR model
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

        prevalence = np.zeros(len(timearray))
        incidence = np.zeros(len(timearray))

        # Timestep 0 is assumed to be when seed cases happen:
        prevalence[0] = np.sum((df_pop['status'] == "I"))
        incidence[0] = np.sum(((df_pop['status'] == "I") & (df_pop['timestep_infected'] == 0)))

        # Then loop from timestep 1 onward:
        for it, t in zip(range(1, len(timearray)), timearray[1:]): # it is integer time index, t is clock time, such that t = it*dt. 
            infected = df_pop[df_pop['status'] == "I"]['id']
            susceptible = df_pop[df_pop['status'] == "S"]['id']
            for i in infected: # TODO test parallelising this loop, not caring about double infections of same infectee in multiple chains
                # Find some others to infect
                N_contacts = int(self.rng.gamma(self.gamma_numcontacts_k, self.gamma_numcontacts_scale, size=1))
                # print("infector i = ", i, ", N_contacts =", N_contacts)
                prob_infectious_contact = 1 - np.exp(-beta*self.dt)
                N_contacts_infected = self.rng.binomial( N_contacts, prob_infectious_contact)
                N_contacts_infected = min(len(susceptible), N_contacts_infected)
                infectees = self.rng.choice(susceptible, size = N_contacts_infected, replace = False)
                df_pop.loc[infectees, 'status'] = "I"
                df_pop.loc[infectees, 'timestep_infected'] = it
                df_pop.loc[infectees, 'timestep_recovered'] = it + (self.rng.gamma(self.gamma_recoverytime_k, self.gamma_recoverytime_scale, size=N_contacts_infected)/self.dt).astype(int) # Deciding recovery time in the future for each individual

            # Who recovers at this timestep?
            recovered = df_pop[df_pop['timestep_recovered'] == it]['id']
            df_pop.loc[recovered, 'status'] = "R"

            # Update counters at this timestep
            prevalence[it] = np.sum((df_pop['status'] == "I"))
            incidence[it] = np.sum(((df_pop['status'] == "I") & (df_pop['timestep_infected'] == it)))
                


        return timearray, prevalence, incidence



if __name__ == "__main__":


    # Instantiate model:
    abm = ABM(
        N_pop=100,
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

    # Plot results
    f, ax = plt.subplots(2)
    ax[0].plot(timearray, incidence, label='incidence')
    ax[0].set_title('incidence')
    ax[1].plot(timearray, prevalence, label='prevalence')
    ax[1].set_title('prevalence')
    plt.tight_layout()
    
    plt.show()