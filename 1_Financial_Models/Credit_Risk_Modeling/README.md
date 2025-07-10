# Credit Risk Modeling

This directory contains implementations of various models and instruments related to credit risk.

## Structural Models
Models that link the probability of default to the firm's asset value and capital structure.

- `merton_model.py`: Implementation of the Merton model for corporate default.
- `black_cox_model.py`: Placeholder for the Black-Cox model, extending Merton with a default barrier.

## Reduced-Form Models
Models that treat default as an exogenous event, often modeled as a stochastic process.

- `jarrow_turnbull_model.py`: A simplified implementation of a Jarrow-Turnbull-like model, treating default as a Poisson process.

## Credit Derivatives
Implementations for pricing and understanding credit-related financial instruments.

- `credit_default_swap.py`: Basic pricing functions for Credit Default Swaps (CDS).

## Probability of Default (PD) Estimation
Models and techniques for estimating the likelihood of a borrower defaulting on their obligations.

- `pd_estimation.py`: Implementation of Probability of Default (PD) estimation using Logistic Regression.
