# Simple infectious disease models written in python

Each model can be run by `python <script_name.py>`. The stochastic models can also be run in ensemble through a wrapper.

## Agent-based model
Found under folder `agent_based/`. The simplest possible implementation of an agent-based SIR model, assuming only random mixing between individuals. The population is stored as a Pandas dataframe and can easily be expanded with further characteristics. Very slow to run.

## Compartmental models
Found under folder `compartmental/`. 

### Deterministic
SIR model based on numerically solving coupled differential equations.

### Stochastic
SIR model based on numerically running a set of coupled difference equations with Bernoulli draws of the number of new infections and recoveries at each timestep.
