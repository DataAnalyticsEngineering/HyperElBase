# HyperElBase

## Short description
GAMM AG DATA collaborative repository for hyperelastic strain energy functions to build benchmarks


## Scope and background
At the 13th GAMM AG Data workshop in Darmstadt (Feb 11/12, 2025), the creation of a set of baseline models for the use of benchmarks and simulations was decided on. This repository collects different models for implementation in `pytorch` or `TensorFlow`. A provisional interface is teased to allow the use from `TensorFlow` as well as from `pytorch`, independent of the implementation (not all features will be accessible though).

A first demonstration for how to use the models is presented in the base class `torch_HypEl` for models written for `pytorch`. It contains also a unified demonstration sub-routine and some utilities, e.g., to generate random deformation tensors or relative errors. The demo sub-routine generalizes to all child classes of `torch_HypEl`, as long as they provide an analytical expression of the first Piola stress through `stress(self, x)`.

## Notation and code design
The notation for the inputs/outputs is as follows:
* An orthonormal coordinate system is asserted throughout for simplicity.
* The input is `x[i, j, k] = F[i][j, k]`, i.e., `x[i]` is the $i$th deformation input, and the trailing 2 indices denote the component of the tensor according to</br>
  $`\boldsymbol{F}_{(i)} = \left( \begin{array}{ccc} F_{11} & F_{12} & F_{13} \\ F_{21} & F_{22} & F_{23} \\ F_{31} & F_{32} & F_{33} \end{array} \right)_{(i)}`$
* The `forward(self, x)` function provides the strain energy `w[i]=`$`W(\boldsymbol{F}_{(i)})`$.
* The [optional] `stress(self, x)` function is used to validate the stresses $`\boldsymbol{P}_{(i)}=\partial_{\boldsymbol{F}} W(\boldsymbol{F}_{(i)})`$; this function can contain a closed-form implementation of the Piola stress.

## Documentation of the models
The documentation will be placed in a LaTeX document for now. This document (including a compiled version thereof) will be added soon.

## Utility routines
Utilities are provided to facilitate testing etc.

* `rel_err(a, a_ref, eps)` computes relative errors and is safe w.r.t. division by zero errors, as long as $`\epsilon>0`$
* `RandomDeformations(n, amp)` defines a dataset of $n$ deformations defined via</br>
  $` \boldsymbol{F}_{(i)} = \left( \begin{array}{ccc} 1 + X & X & X \\ X & 1+X & X \\ X & X & 1+X \end{array} \right)_{(i)},`$</br>
  where $`X \sim `$ `amp * `$`\mathcal{U}([-1, 1])`$ (i.e. uniformly random entries of the displacement gradient).
* `demo_HypEl(HypElModel, n, amp)` uses the model `HypElModel` and computes the stresses using auto-differentiation (via `autograd_stress(self, x)` from the base class `torch_HypEl`). They are compared to the analytical stress from `stress(self, x)` and the relative errors are printed for each sample.
* `demo_NeoHooke()` illustrates the use of `demo_HypEl` for a Neo Hooke model.
