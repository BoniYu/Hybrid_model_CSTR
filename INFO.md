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

# Info: Concepts Behind the Hybrid CSTR Model

A reference for the ideas used in this project. Read it before the code, or whenever a term is unclear. It uses the same examples as the tutorial (MLCE_book, tutorial 5) and the same names as the notebook.

## The project in one paragraph

A CSTR (continuous stirred tank reactor) runs the reaction A + B ⇌ X. Concentrations were measured over time. The mass balances are known from physics (an **ODE**), but the reaction rate is not. A small neural network learns the rate. To train it, the model needs gradients of the loss with respect to the network weights, which come from **sensitivity equations** and **automatic differentiation**.

## Glossary

| Term | Meaning |
|---|---|
| Hybrid (grey-box) model | A model that combines known physics with machine learning for the unknown part |
| ODE | An equation that relates a quantity to its own rate of change, solved to get the quantity over time |
| Derivative | The slope of a function at a point |
| Jacobian | A table of derivatives: one row per output, one column per input |
| Sensitivity | How much a model output changes when a parameter changes slightly |
| Loss (objective) | One number measuring how far the predictions are from the measurements |
| Inlet concentration | Concentration in the feed stream entering the reactor (constant) |
| Initial concentration | Concentration inside the reactor at time 0 (the starting point for solving) |
| Residence time τ | Reactor volume divided by flow rate (100 s in the tutorial) |

---

## 1. Automatic differentiation

### What it is

A way for the computer to calculate exact derivatives of a function written as ordinary code. There are three ways to get a derivative:

- **Numerical (finite differences):** compute (f(x+h) − f(x)) / h. It is approximate and depends on the choice of h.
- **Symbolic:** manipulate the formula algebraically. It is exact, but the expressions can become very large.
- **Automatic:** while the function runs, record every elementary step (add, multiply, exp, ...) and apply the chain rule through the record. It is exact to floating-point precision and cheap.

### Worked example

f(x) = (3x + 1)² at x = 2. Split it into steps: a = 3x + 1 = 7 (da/dx = 3), then f = a² = 49 (df/da = 2a = 14). By the chain rule, df/dx = 14 × 3 = 42. Automatic differentiation does exactly this bookkeeping for you.

### In code (the `autograd` library)

```python
import autograd
import autograd.numpy as np   # use this NumPy, not plain numpy

def f(x):
    return (3.*x + 1.)**2

df = autograd.grad(f)         # returns a NEW function
print(df(2.))                 # 42.0
```

Rules to remember:
- `autograd.grad(f)` returns a function, not a number.
- Inputs must be **floats** (`2.`, not `2`), or you get a `TypeError`.
- `grad` needs a function that returns **one number**.
- Use `autograd.numpy` as `np`, so every operation is recorded.
- To differentiate with respect to another argument, pass its position (counting from 0): `autograd.grad(p, 1)`.

### Jacobians: functions that return several numbers

`autograd.jacobian` handles functions with several outputs. Row i, column j holds ∂(output i)/∂(input j). The shape of the result is the output shape followed by the input shape.

A 0 entry means "this output does not depend on this input".

### Where it is used here

```python
dfdc = autograd.jacobian(dcdt, 0)   # derivative of the balances w.r.t. the concentrations
dfdp = autograd.jacobian(dcdt, 2)   # derivative of the balances w.r.t. the network parameters
```

`dcdt(c, t, params)` has arguments at positions 0 (`c`), 1 (`t`) and 2 (`params`). The shapes are (3, 1, 3) and (3, 1, 16): three equations, one network output, and 3 concentrations or 16 parameters.

---

## 2. ODE solving

### Derivative versus ODE

- **Derivative:** you know the function and want its slope. Given c(t) = e^(−kt), the slope is −k·e^(−kt).
- **ODE:** you know a rule for the slope and want the function. Given dc/dt = −k·c, find c(t).

Autograd differentiates. A solver (`odeint`) solves.

### How a solver works (idea)

It starts at the initial value and walks forward in small steps, using the slope at each point: new value ≈ old value + step × slope. Real solvers choose very small steps automatically and use better formulas.

### In code

```python
from scipy.integrate import odeint

def decay(y, t, k):          # the rule: returns the rate of change of each variable
    return -k * y

sol = odeint(decay, [1.0], t, args=(0.5,))
```

| Input | Meaning |
|---|---|
| function | takes `(y, t, extras...)` and returns dy/dt, with the same length as `y` |
| `y0` | the starting values |
| `t` | the times at which you want the answer |
| `args` | a tuple of the extra arguments (note the trailing comma) |

### What `sol` is

A table of shape (number of times, number of variables). Each **row** is one time point, and each **column** is one variable. The first row is `y0`. Values are given only at the times in `t`; the solver's own small steps in between are not returned.

### In this project

`odeint(DiffEqs, y0, t_span, args=(params,))` solves a combined system of **51 variables**: 3 concentrations plus 48 sensitivities (3 species × 16 parameters). So `sol` has shape (30, 51), with 30 time points.

| Selection | Gives |
|---|---|
| `sol[:, :3]` | all rows, first 3 columns: the predicted concentrations |
| `sol[:, 3:]` | all rows, columns 3 onwards: the sensitivities |
| `sol[-1, :3]` | the concentrations at the final time |

Note that `sol[3:]` (without a comma) slices **rows**, not columns.

---

## 3. Sensitivity equations

### What a sensitivity is

For a model with parameter θ, the sensitivity s = ∂y/∂θ says how much the output y changes when θ changes slightly. For an ODE model, it is a curve over time.

### What the sensitivity equation is

The sensitivity obeys an ODE of its own. For a model dy/dt = f(y, θ):

ds/dt = (∂f/∂y) · s + ∂f/∂θ, with s(0) = ∂y(0)/∂θ

- **(∂f/∂y) · s:** an existing change is carried forward by the model's own dynamics.
- **∂f/∂θ:** the parameter also affects the rate directly.

If the starting values do not depend on the parameter, s(0) = 0.

### Toy example

Take dc/dt = −k·c with c(0) = 1. Differentiating with respect to k (the product rule applies, because c also depends on k) gives:

ds/dt = −k·s − c, with s(0) = 0

The solution is s(t) = −t·e^(−kt), which is also what you get by differentiating c(t) = e^(−kt) directly. It is negative because a larger k means faster decay and a lower c.

### Why they are used in this project

The loss depends on the concentrations, and the concentrations come from an ODE solver that autograd cannot differentiate through. No measurements of the rate r exist to train the network directly. The sensitivities ∂C/∂w give the missing link:

- gradient of the loss for weight j = sum over time points and species of (prediction − measurement) × sensitivity

For the reactor, the sensitivity equations (one per species and per parameter) are:

d/dt [∂C/∂w] = (∂f/∂C) · [∂C/∂w] + ∂f/∂w

The matrix (∂f/∂C) is `dfdc`. The vector ∂f/∂w is `dfdp`. Both come from autograd. `DiffEqs` stacks the concentrations and the sensitivities into one state vector, and `odeint` integrates them together in a single run.

State vector layout, for the tutorial network `[3, 3, 1]` (16 parameters):

| Positions in y | Contents |
|---|---|
| 0 to 2 | C_A, C_B, C_X |
| 3 to 5 | ∂C_A/∂w1, ∂C_B/∂w1, ∂C_X/∂w1 |
| 6 to 8 | the same three for w2 |
| ... | ... |
| 48 to 50 | the same three for w16 |

### Cost

For p parameters and l states there are p × l extra equations. The tutorial network has 16 parameters and 3 states, so 48 extra equations. A 3-10-1 network would have 51 parameters and need 153. This is why larger models often use other methods.

### Other uses of sensitivity equations

- fitting kinetic parameters (rate constants, activation energies) to data
- uncertainty and identifiability (they feed the Fisher information matrix)
- experiment design (choosing experiments where the output is most sensitive to the parameters)
- process analysis (which parameters matter most for an output)
- optimal control

### Alternatives

- **Autodiff through the solver** (for example with PyTorch and `torchdiffeq`, or JAX and `diffrax`) avoids writing sensitivity equations by hand.
- **The adjoint method** is usually cheaper when there are many parameters and one scalar loss.

---

## How the three fit together

```
weights → neural network → rate r → ODE solver → predicted concentrations
   ↑                                                      │
   │                                                      ▼
update weights ← gradient (via sensitivities) ← loss ← compare with data
```

- The **ODE solver** turns rates into concentration curves.
- **Automatic differentiation** supplies the derivatives (Jacobians) used inside the sensitivity equations.
- The **sensitivity equations** turn those derivatives into gradients for training.

## Notes on the tutorial code

- The tutorial text says autograd is "the implementation of autodiff in the pytorch package". The code actually uses the standalone `autograd` library, not PyTorch.
- `objective` returns the plain sum of squared errors, while `objective_grad` returns the gradient of half of that sum. The factor of 2 is harmless for the Adam optimiser.
- `neural_net_predict` resets its position marker `sep` on each layer, which is correct only for networks with one hidden layer, such as `[3, 3, 1]`.