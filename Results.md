# Results and Learnings: Hybrid Model for a CSTR

Summary of what I built, what I found, and what I learned. This project follows tutorial 5 of the MLCE_book ("Hybrid model for CSTR"). All of the modelling approach comes from that tutorial. The learning path, the explanations in `info.md` and `Steps.md`, and the notes below are my own work.

## What the project does

A neural network learns the unknown reaction rate inside a mechanistic ODE model of a CSTR (reaction A + B ⇌ X). The network is trained on concentration measurements using sensitivity equations and automatic differentiation.

## Steps taken

### Learning path (before the main model)

1. Learned automatic differentiation with `autograd`: `grad`, differentiating with respect to a chosen argument, higher derivatives, and `jacobian`.
2. Computed the Jacobian of the CSTR balance equations and checked each entry by hand.
3. Trained a one-neuron model, then a two-input model, by gradient descent using `autograd` gradients.
4. Worked through the idea of sensitivities with a toy decay ODE, dc/dt = −k·c, and derived part of the sensitivity matrix by hand.
5. Learned how `odeint` works and how its output (`sol`) is laid out.

### The model (following the tutorial)

| Step | What happens | Code |
|---|---|---|
| 1 | Write the reactor mass balances (the ODE) | `dcdt` |
| 2 | Replace the rate with a neural network (3 inputs, 3 hidden nodes, 1 output; 16 parameters) | `init_random_params`, `neural_net_predict` |
| 3 | Solve the ODE forward in time | `odeint` |
| 4 | Compare predictions with measured concentrations (sum of squared errors) | `objective` |
| 5 | Integrate 48 sensitivity equations alongside the concentrations; combine them with the errors to get gradients | `DiffEqs`, `objective_grad` |
| 6 | Update the weights with Adam, 1000 iterations | `adam` |
| 7 | Plot concentrations before and after training against the data | plotting cell |

## Results

- Loss at the start of training: TODO (add your value)
- Loss at the end of training: TODO (add your value)
- For reference, the tutorial reports a loss of about 4.28 at step 0 and about 2.4e-2 at step 990. Check whether your run matches.
- Plot: TODO (save the figure to `figures/` and link it here, for example `![Concentrations before and after training](figures/trajectories.png)`)

What the plot shows: the untrained model (dashed lines) is far from the measurements, and the trained model (solid lines) follows them closely. TODO: confirm this matches your own plot before keeping this sentence.

## What I learned

### Technical

- **Automatic differentiation** gives exact derivatives of functions written as code, by recording each step and applying the chain rule. It differs from numerical differentiation (approximate) and symbolic differentiation (can become very large).
- **Derivative versus ODE:** differentiating means going from a function to its slope; solving an ODE means going from a rule for the slope to the function.
- **Sensitivity equations** give ∂(output)/∂(parameter) for an ODE model by integrating one extra ODE per parameter and species. They are what lets a neural network inside an ODE be trained when the quantity it predicts (the rate) is never measured.
- **Why they are used here:** the ODE solver is a black box to `autograd`, so the gradient has to be built from sensitivities. The cost grows as parameters × states, which becomes heavy for larger networks.
- **Hybrid modelling** puts physics where it is known and machine learning where it isn't. The network only needs to be differentiable, so other models (for example a fitted rate law) could take its place.
- **Starting weights matter:** the network's output feeds the ODE directly, so starting weights are scaled to be tiny. Otherwise the initial rate is far too large and the first solves go wrong.
- **Input size and parameter count affect training speed:** with the same learning rate, a model with larger inputs and more parameters moved much faster in a small experiment.

### Practical

- Autograd needs float inputs and the `autograd.numpy` wrapper. Integers and plain lists caused errors.
- Many bugs showed up as numbers that looked plausible but were wrong. Checking a result by hand, or checking its sign and shape, caught them.
- Array shapes carry meaning here (for example (3, 1, 16) for the parameter Jacobian). Reading shapes first made dense functions easier to follow.
- Slicing details matter: `sol[:, :3]` selects columns, while `sol[3:]` selects rows.
- Reading code I didn't write went best one function at a time, with a small worked example for each confusing line.

### About the tutorial code

- Several functions combine many ideas in a few lines (`DiffEqs`, `objective_grad`), which made them hard to follow.
- `objective` returns the plain sum of squared errors, while `objective_grad` returns the gradient of half of that sum. The factor of 2 is harmless with Adam. TODO: confirm this on your own data if you want to quote it.
- The network's parameter-slicing code is correct only for one hidden layer.

## Limitations

- This replicates a published tutorial on its provided data. It is not an independent study.
- One small network, one dataset, one training run. No noisy-data test and no test of predictions outside the training conditions.
- The network has no advantage here over a fitted power-law rate. The method matters most when the form of the rate is unknown.

## Next steps

- [ ] Rebuild the project on a new branch with one idea per function, and compare against the tutorial results
- [ ] Redo it in PyTorch (`torchdiffeq`) and compare
- [ ] Compare the network with a fitted power-law rate, on clean and noisy data