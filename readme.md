# Simple infectious disease models written in python

## Agent-based model
Found under folder `agent_based/`. The simplest possible implementation, assuming only random mixing between individuals. The population is stored as a Pandas dataframe and can easily be expanded with further characteristics. Very slow to run.

## Compartmental models
Found under folder `compartmental/`. 

### Deterministic
SIR model based on numerically solving coupled differential equations.

### Stochastic
SIR model based on numerically running a set of coupled difference equations with Bernoulli draws of the number of new infections and recoveries at each timestep.
