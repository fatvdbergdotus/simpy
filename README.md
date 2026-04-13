# A Simpy implementation of chapter 4 of my PhD thesis
[Jupyter Notebook](phd_thesis/09%20PhD%20thesis%20iDSL.ipynb)

A SimPy simulation of a Philips Imaging System with two concurrent systems, the so-called Biplane system, for 3D images.

Both systems need to repeat for shared resources and what can be seen from different simulation runs is that when the different imaging systems are further apart in time, less concurrency happens.

A CDF for different times is the result.

See also [https://drops.dagstuhl.de/entities/document/10.4230/OASIcs.MCPS.2014.80](https://drops.dagstuhl.de/entities/document/10.4230/OASIcs.MCPS.2014.80).

# A Simpy/Z3 implementation of a fictional printer case
[Python executable](printer_costs/compute_cost_ecdfs.py)

A SimPy/z3 implementation of a fictional printer cases. The printer consists of multiple parts, which are prices either non-determinsitcally or probabilistically.

Monte Carlo simulation is used to deal with the probabilistic part by running many Z3 solver computations with different inputs for the probabilistic prices.

A CDF for the lower bound and one for the upper bound is the result.
