# approximate bayesian computation struggles
## simulated annealing approach: 
based off of https://royalsocietypublishing.org/doi/10.1098/rspa.2018.0129

uses simulated annealing to infer parameters for the start of the contagion; 
doesn't do super great but it works! it runs!

seen in misinfo_simulation.py

## approximate bayesian computation approach: 
https://arxiv.org/pdf/1208.2157.pdf

based off the above paper. uses ABC based somewhat on simulated annealing to infer a distribution over parameters for the start of the contagion. found in misinfo_abc.py. 

a more commented copy of the same code is found in bayesian_things.ipynb along with some plots of the parameter distributions
## utilities etc
misinfo_functions.py contains functions for computing relevant quantities and simulations

utilities.py contains utility functions for log computation etc

agents.py instantiates the agent classes, which are hopefully flexible enough for our purposes.
