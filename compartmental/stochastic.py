import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numba

"""
Stochastic comparmental SIR model
By Jørgen Eriksson Midtbø, Folkehelseinstituttet, github.com/jorgenem
"""

class SIR:
    """
    Simulate the following system of difference equations, which is analogous to the deterministic SIR model except the "derivatives" are replaced by binomial draws using exponential probability distributions:
    # First calculate probabilities and stochastic changes at each timestep
    p_SI = 1 - exp(-beta * I[i] / N * dt)
    n_SI = binomial(S[i], p_SI)
    p_IR = 1 - exp(-gamma * dt)
    n_IR = binomial(I[i], p_IR)
    # Then update values
    S[i+1] = S[i] - n_SI
    I[i+1] = I[i] + n_SI - n_IR
    R[i+1] = R[i] + n_IR
    

    Note that S, I and R are "unitful", i.e. they sum to the total population Npop and not 1
    """
    def __init__(self, T, dt, beta, gamma, N, I0, R0=0, random_seed=None):
        self.T = T
        self.dt = dt # Only used for plotting, the solver decides its own timestep
        self.beta = beta 
        self.gamma = gamma 
        self.N = N
        self.I0 = I0
        self.R0 = R0
        self.S0 = N - I0 - R0

        # Initialise random number generator
        self.rng = np.random.default_rng(seed=random_seed)

    def rhs(self, y):
        """"Right-hand side of the equation set. 
        Args:
            y = [S[i], I[i], R[i]]
        Returns:
            [S[i+1], I[i+1], R[i+1]]

        """
        p_SI = 1 - np.exp(-self.beta * y[1] / self.N * self.dt)
        p_IR = 1 - np.exp(-self.gamma * self.dt)
        n_SI = self.rng.binomial(y[0], p_SI)
        n_IR = self.rng.binomial(y[1], p_IR)
        ynew = np.array([
            y[0] - n_SI,
            y[1] + n_SI - n_IR,
            y[2] + n_IR
        ])

        return ynew


    # @numba.jit(nopython=True) # TODO Speed up by numba just-in-time compilation of code? Seems not to work inside class structure
    def run_model(self):
        t_array = np.arange(0, self.T, self.dt)
        y = np.zeros((len(t_array), 3)) # Store arrays of [S, I, R] timeseries
        y[0, :] = np.array([self.S0, self.I0, self.R0])
        for i in range(len(t_array)-1):
            y[i+1, :] = self.rhs(y[i, :])

        return t_array, y[:, 0], y[:, 1], y[:, 2] # Returning t, S, I, R as separate arrays


if __name__ == "__main__":

    # Set parameters and initial conditions
    T = 80
    dt = 0.1
    beta = 1.4
    gamma = 0.8
    N = 100
    I0 = 1
    R0 = 0

    # Run model
    sir = SIR(T=T, dt=dt, beta=beta, gamma=gamma, N=N, I0=I0, R0=R0)
    t, S, I, R, = sir.run_model()


    ## Plot results
    # Plot prevalence curves
    f, ax = plt.subplots(1)
    ax.plot(t, S, label="S")
    ax.plot(t, I, label="I")
    ax.plot(t, R, label="R")
    ax.legend()

    plt.show()