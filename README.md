# Physics-Informed Neural Networks (PINNs) in Julia 

PINNs are **coordinate networks** trained to be the solution to an
initial-boundary value problem. The optimization problem is based on minimizing
the residuum of the PDE in the domain, as well as residuums on the boundary and
initial conditions. Additional supervised data (i.e., labeled point-wise
solution values) are optional. The classical approach to PINNs is to use the
automatic differentiation capabilities of deep learning frameworks to compute
the input-output derivatives of the network. For a second-order PDE, that
requires three hierarchical autodiff passes (two to obtain the residuum loss and
a final one to backpropagate into the parameter space). As of the creation of
this repo, Julia does not yet properly support higher-order autodiff, so we
implement the input-output derivatives of the network manually. For a simple
MLP, this is still manageable.

This repository contains a simple PINN example for the 1d Poisson equation with
homogeneous Dirichlet boundary conditions. Check out the intro part of the
Jupyter notebook for more details. You might also find the accompanying [YouTube
video](https://youtu.be/Xfb7tqs7gQA) helpful to code along.

**Generally interested in Scientific Machine Learning?** Check out my [YouTube
channel](https://www.youtube.com/@machinelearningsimulation) and the
corresponding [GitHub
repository](https://github.com/Ceyron/machine-learning-and-simulation).
