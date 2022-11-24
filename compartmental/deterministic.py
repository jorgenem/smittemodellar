from xml.sax.handler import DTDHandler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp

"""
A simple deterministic SIR model
By Jørgen Eriksson Midtbø, Folkehelseinstituttet, github.com/jorgenem
"""

class SIR:
    """
    Solve the following system of differential equations:
    dS/dt = -beta*S*I/N
    dI/dt = beta*S*I/N - gamma*I
    dR/dt = gamma*I

    Note that S, I and R are "unitful", i.e. they sum to the total population Npop and not 1
    """
    def __init__(self, T, dt, beta, gamma, N, I0, R0=0):
        self.T = T
        self.dt = dt # Only used for plotting, the solver decides its own timestep
        self.beta = beta 
        self.gamma = gamma 
        self.N = N
        self.I0 = I0
        self.R0 = R0
        self.S0 = N - I0 - R0

    def f(self, t, y):
        """"Right-hand side of the equation set. y = [S, I, R]"""
        return [
            -self.beta * y[0] * y[1] / self.N,
            self.beta * y[0] * y[1] / self.N - self.gamma * y[1],
            self.gamma * y[1]
        ] 
    
    def run_model(self):
        sol = solve_ivp(
            self.f, 
            t_span=[0, self.T], 
            y0=[self.S0, self.I0, self.R0], 
            t_eval=np.arange(0, self.T, self.dt)
        )
        t = sol.t
        S, I, R = sol.y
        return t, S, I, R




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
    f, ax = plt.subplots(2)
    ax[1].plot(t, S, label="S")
    ax[1].plot(t, I, label="I")
    ax[1].plot(t, R, label="R")
    ax[1].legend()

    # Plot incidence of infections (has to be calculated explicitly from the first term in dI/dt since compartmental value is prevalence)
    inc = beta * S * I / N
    ax[0].plot(t, inc, label="incidence I")
    ax[0].legend()

    plt.show()
    