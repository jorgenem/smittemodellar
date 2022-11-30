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
    T = 20
    dt = 0.1
    beta = 1.5
    gamma = 0.4
    N = 1000
    I0 = 10
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
    ax.set_ylabel("Prevalence")
    ax.set_xlabel("Time")
    ax.legend()

    # Plot incidence of infections (has to be calculated explicitly from the first term in dI/dt since compartmental value is prevalence)
    inc = beta * S * I / N
    f, ax = plt.subplots(1)
    ax.plot(t, inc, label="incidence I")
    ax.legend()

    plt.show()


    # Plot "phase diagrams"
    f, ax = plt.subplots(3)
    ax[0].plot(S, I, label="SI")
    ax[1].plot(I, R, label="IR")
    ax[2].plot(R, S, label="RS")
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
