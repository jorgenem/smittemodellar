# Simple infectious disease models written in python

## Agent-based model
Found under folder `agent_based/`. The simplest possible implementation, assuming only random mixing between individuals. The population is stored as a Pandas dataframe and can easily be expanded with further characteristics. Very slow to run.

## Deterministic compartmental model
SIR model based on numerically solving coupled differential equations.

## Stochastic compartmental model
SIR model based on numerically running a set of coupled difference equations with Bernoulli draws of the number of new infections and recoveries at each timestep.
