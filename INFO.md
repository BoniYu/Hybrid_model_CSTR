# Steps: How the Hybrid CSTR Model Works

A plain-language walkthrough of the project, from problem to result. It follows tutorial 5 of the MLCE_book ("Hybrid model for CSTR") and maps each step to the code.

## The situation

A CSTR (continuous stirred tank reactor) runs the liquid-phase reaction A + B ⇌ X. Concentrations of A, B and X were measured over time. The mass balances are known from physics, but the **reaction rate r** is treated as unknown.

## The idea

Use physics for what is known (the ODE mass balances) and let a neural network **learn the unknown part** (the reaction rate). This combination is a **hybrid (grey-box) model**.

## Pipeline at a glance

```
weights → neural network → rate r → ODE solver → predicted concentrations
   ↑                                                      │
   │                                                      ▼
update weights ← gradient (via sensitivities) ← loss ← compare with measured data
```

## Step 1: Write the physics (the ODE)

Mass balance for each species (τ = V/F is the residence time, 100 s in the tutorial):

- dC_A/dt = (C_A,in − C_A)/τ − r
- dC_B/dt = (C_B,in − C_B)/τ − r
- dC_X/dt = (C_X,in − C_X)/τ + r

Each equation has a feed/outflow part and a reaction part. A and B are consumed (−r), X is produced (+r).

- **Inlet concentrations** (constant feed): C_A,in = 0.7, C_B,in = 0.3, C_X,in = 0
- **Initial concentrations** (what is in the reactor at t = 0): C_A,0 = 0.5, C_B,0 = 0.5, C_X,0 = 0

The inlet values are used inside the equations. The initial values are only the starting point for solving.

**Role of the ODE:** the physics skeleton that turns rates into concentration curves.
**In code:** `dcdt`

## Step 2: Put a neural network in place of r

The network takes the three concentrations and returns the reaction rate: r = NN(C_A, C_B, C_X).

- Tutorial setup: 3 input nodes, 3 hidden nodes, 1 output node (`layer_sizes = [3, 3, 1]`)
- Parameter count: 3·3 + 3 + 3·1 + 1 = 16 weights and biases, stored in one flat array
- Weights start random and tiny (scaled by 0.0005), so the starting rate is close to zero and the first ODE solves stay well behaved
- The output layer is linear (no sigmoid), so r is not forced into the range 0 to 1

**Role of the neural network:** it produces the unknown rate, and it is called every time the ODE solver needs r.
**In code:** `init_random_params`, `neural_net_predict`

## Step 3: Solve the ODE forward in time

Starting from the initial concentrations, the ODE solver steps through time using the network's current rate. The result is the **predicted** concentration curves for A, B and X.

- Time span in the tutorial: 0 to 100 s, 30 points

**Role of the ODE solver:** turns the rate into concentrations over time.
**In code:** `odeint(DiffEqs, y0, t_span, args=(params,))`

## Step 4: Compare with the measurements (the loss)

Compare the predicted curves with the measured data. The loss is the sum of squared differences over all time points and all three species.

Squared errors are used so that over- and under-prediction do not cancel, bigger errors cost more, and the loss is smooth, which gradients need.

**In code:** `objective`

## Step 5: Work out how to improve the weights (sensitivities)

To update a weight, we need to know how the loss changes when that weight changes. The loss depends on the concentrations, and the concentrations depend on the weights through the ODE. The missing link is the **sensitivity**:

- s = ∂C/∂w: how much a concentration changes when a weight changes slightly

Sensitivities are time-dependent, so each one has its own ODE. They are integrated together with the concentrations, in one run:

- d s/dt = (∂f/∂C) · s + ∂f/∂w

where f is the right-hand side of the balances. The two ingredients are Jacobians that **autograd** computes:

- `dfdc = autograd.jacobian(dcdt, 0)`: derivative with respect to the concentrations
- `dfdp = autograd.jacobian(dcdt, 2)`: derivative with respect to the network parameters

Then the gradient of the loss for each weight is the sum of (prediction − measurement) × sensitivity.

- Count: 3 species × 16 parameters = 48 sensitivity equations, plus the 3 concentration equations
- Cost: grows quickly with network size (a 3-10-1 network would need 153 sensitivities)

**In code:** `DiffEqs`, `objective_grad`, `dfdc`, `dfdp`

## Step 6: Update the weights and repeat

An optimiser (Adam) uses the gradient to adjust the weights. Then Steps 3 to 5 repeat.

- Tutorial settings: 1000 iterations, step size 0.001
- The loss should fall steadily as the rate becomes more accurate

**In code:** `adam(objective_grad, init_params, ...)`

## Step 7: Check the result

Plot the concentration curves before training, after training, and the measured data. After training, the curves should lie close to the data.

**In code:** the final plotting cell

## Quick map: step to code

| Step | What happens | Code |
|---|---|---|
| 1 | Physics (ODE balances) | `dcdt` |
| 2 | Neural network for the rate | `init_random_params`, `neural_net_predict` |
| 3 | Solve the ODE forward | `odeint` |
| 4 | Loss against measured data | `objective` |
| 5 | Sensitivities and gradient | `DiffEqs`, `objective_grad` |
| 6 | Update the weights, repeat | `adam` |
| 7 | Plot before and after | plotting cell |

## Glossary

- **Hybrid model:** a model that combines physics (known parts) with machine learning (unknown parts)
- **ODE:** an equation involving a derivative with respect to time, solved to get a curve over time
- **Autograd:** automatic differentiation, which computes exact derivatives of functions written as code
- **Jacobian:** a table of derivatives, with one row per output and one column per input
- **Sensitivity:** how much a concentration changes when a parameter changes slightly
- **Loss:** a single number measuring how far the predictions are from the data

## Next steps

- [ ] Redo the network with a cleaner layout (list of weight/bias pairs plus autograd's `flatten`) and compare with the tutorial results
- [ ] Re-implement in PyTorch (torchdiffeq) and compare