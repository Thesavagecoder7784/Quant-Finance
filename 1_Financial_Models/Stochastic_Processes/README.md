# Stochastic Processes in Finance

This directory contains Python implementations for simulating various fundamental stochastic processes, which are crucial for modeling financial phenomena that evolve randomly over time.

## Contents

- **branching_process.py:**
  Simulation of a branching process, often used to model population growth or the spread of information/defaults in a financial network.

- **brownian_motion_simulation.py:**
  Simulation of a standard Brownian Motion (Wiener Process), a fundamental continuous-time stochastic process used in option pricing and risk modeling.

- **continuous_time_markov_chain.py:**
  Implementation of a continuous-time Markov chain, useful for modeling credit ratings transitions or state changes in financial systems.

- **discrete_time_markov_chain.py:**
  Implementation of a discrete-time Markov chain, used for modeling sequences of events or states in financial markets.

- **gaussian_process.py:**
  Demonstration of a Gaussian Process, a powerful non-parametric model used for regression and classification, with applications in time series forecasting and volatility modeling.

- **geometric_random_walk.py:**
  Simulation of a geometric random walk, often used as a simplified model for asset prices, where price changes are proportional to the current price.

- **jump_diffusion.py:**
  Simulation of a jump-diffusion process, which combines continuous diffusion with sudden, discrete jumps, useful for modeling assets with occasional large price movements.

- **levy_process.py:**
  Implementation of a Lévy process, a generalization of Brownian motion and Poisson processes, allowing for more complex jump behaviors in financial modeling.

- **martingale_process.py:**
  Demonstration of a martingale process, a sequence of random variables where the expected value of the next variable, given all prior variables, is equal to the current variable. Crucial in arbitrage-free pricing.

- **poisson_process_simulation.py:**
  Simulation of a Poisson process, used to model the occurrence of discrete events over time, such as the arrival of trades or defaults.

- **renewal_process.py:**
  Implementation of a renewal process, which models the number of events that occur over time, where the time between events is independent and identically distributed.

- **simple_random_walk.py:**
  Simulation of a simple random walk, a basic discrete-time stochastic process often used as an introductory model for asset price movements.
