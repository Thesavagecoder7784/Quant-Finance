# Stochastic Processes Simulation
Provides Python implementations for simulating various fundamental stochastic processes from scratch. 

## Simulations
1. Brownian Motion Simulation
2. Poisson Motion Simulation
3. Jump Diffusion Simulation

## 1. Brownian Motion Simulation
This script simulates a standard Brownian Motion (also known as a Wiener Process). It generates a continuous path by accumulating independent, normally distributed increments over time.

### Key Properties:
- Continuous Paths: The simulated paths are continuous, though theoretically non-differentiable.
- Independent Increments: Changes in the process over non-overlapping time intervals are statistically independent.
- Stationary Increments: The distribution of an increment depends only on the length of the time interval, not its starting point.
- Normal Distribution: The value of B(t) at any time t (starting from B(0)=0) is normally distributed with mean 0 and variance t, i.e., B(t)
simN(0,t).

### Applications:
- Option Pricing (Black-Scholes): Forms the basis for modeling asset prices (via Geometric Brownian Motion) in the Black-Scholes option pricing model.
- Risk Modeling: Used to model market risk and asset price volatility.
- Stochastic Calculus: Fundamental building block for Itô calculus and solving Stochastic Differential Equations (SDEs).
- Physics: Describes the random movement of particles (e.g., pollen in water).

## 2. Poisson Motion Simulation
This script simulates a homogeneous Poisson Process. It models the occurrence of discrete events over time, where events happen at a constant average rate (
lambda). The simulation is based on generating inter-arrival times from an exponential distribution.

### Key Properties:
- Discrete Events: Counts the number of discrete events over time.
- Memorylessness: The probability of an event occurring in the next infinitesimal time interval is independent of when the last event occurred.
- Constant Rate: Events occur at a constant average rate (
lambda).
- Poisson Distribution: The number of events N(t) in any interval of length t follows a Poisson distribution with parameter (lambda timest)
- Exponential Inter-arrival Times: The time between consecutive events is exponentially distributed with parameter 
lambda.

### Applications:
- Queueing Theory: Modeling customer arrivals, calls to a call center.
- Insurance: Modeling the arrival of claims.
- Telecommunications: Modeling packet arrivals in networks.
- Finance: Modeling the arrival of rare events like large price jumps or defaults (often as part of more complex processes).

## 3. Jump Diffusion Simulation
This script simulates a Jump Diffusion Process, which combines a continuous diffusion component (like Brownian Motion) with a discrete jump component (modeled by a Compound Poisson Process). This process is particularly useful for modeling phenomena that exhibit both gradual changes and sudden, large movements.

### Key Properties:
- Hybrid Movement: Captures both continuous price movements and sudden, discontinuous jumps.
- Non-Normal Returns: Can generate returns with "fat tails" (leptokurtosis) and skewness, which are commonly observed in financial markets, unlike pure Brownian Motion.
- Discontinuous Paths: The presence of jumps makes the simulated paths discontinuous.

### Applications:
- Option Pricing: Used to price options on assets that are subject to sudden, significant price changes (e.g., due to news, earnings announcements, or market shocks). Merton's (1976) model is a classic example.
- Risk Management: Provides a more realistic framework for assessing extreme events and tail risk (e.g., for Value-at-Risk calculations) compared to models based solely on continuous processes.
- Credit Risk Modeling: Jumps can represent sudden defaults or credit rating downgrades.
- Commodity Markets: Useful for modeling commodity prices that can experience abrupt supply/demand shocks.