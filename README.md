# Hybrid Model for a CSTR

A hybrid (grey-box) model of a continuous stirred tank reactor (CSTR) in which a neural network learns the reaction rate inside a mechanistic ODE mass balance. The network is trained with sensitivity equations and automatic differentiation (autograd).

Based on tutorial 5, "Hybrid model for CSTR", of the [Machine Learning in Chemical Engineering](https://edgarsmdn.github.io/MLCE_book/05_Hybrid_CSTR.html) book (MLCE_book).

> Work in progress. This README will be updated as the project develops.

## Problem

Liquid-phase reaction A + B ⇌ X in a CSTR with constant volume and perfect mixing. The mass balances for C_A, C_B and C_X are known (mechanistic part). The reaction rate r is not assumed, and a neural network r = f(C_A, C_B, C_X) learns it from concentration measurements (data-driven part).

## Plan

- [ ] Learn autograd (`grad`, `jacobian`, differentiating through a small neural network)
- [ ] Build the mechanistic CSTR model
- [ ] Add the neural network for the reaction rate
- [ ] Derive and implement the sensitivity equations
- [ ] Train the hybrid model and compare trajectories before and after training

## Setup

- Python 3.12
- Packages: `autograd`, `numpy`, `scipy`, `pandas`, `matplotlib`, `jupyter`, `ipykernel`

```
python -m venv .venv
.venv\Scripts\activate
pip install autograd numpy scipy pandas matplotlib jupyter ipykernel
```

## Reference

Edgar Ivan Sanchez Medina, Antonio del Rio Chanona and Caroline Ganzer. *Machine Learning in Chemical Engineering* (MLCE_book). Tutorial 5 was initially prepared by Marcus Wenzel.