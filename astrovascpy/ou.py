"""
Copyright (c) 2023-2023 Blue Brain Project/EPFL
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import scipy.optimize as scpo
import scipy.stats as ss
from scipy import sparse
from scipy.sparse.linalg import spsolve


def ornstein_uhlenbeck_process(
    kappa, sigma, dt, iterations=1, zero_noise_time=1, seed=1234
):  # pragma: no cover
    """Monte Carlo simulation of the OU process with a reflecting boundary at zero.

        Stochastic differential equation:
            dX(t) = - kappa X(t) dt + sigma dW
        with solution:
            X(t+dt) = X(t) * exp(-kappa dt) + alpha * epsilon(t)
        where:
        alpha = sqrt( sigma^2 / (2kappa) * (1-exp(-2*kappa*dt)) )
        alpha is the standard deviation of epsilon.
        epsilon is a standard Gaussian random variable.
        The process starts at X(0) = 0

    Args:
        kappa (float): mean reversion coefficient.
        sigma (float): diffusion coefficient of the noise W
        dt (float): time-step.
        iterations (int): number of iterations.
        zero_noise_time (int): index at which the noise is set to zero
                               for the rest of the simulation
        seed (int): RNG seed

    Returns:
        numpy.array: Reflected OU process starting at zero.
    """
    np.random.seed(seed=seed)

    X = np.zeros(iterations + 1)
    alpha = np.sqrt(sigma**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * dt)))
    exp_ = np.exp(-kappa * dt)
    epsilon = ss.norm.rvs(loc=0, scale=alpha, size=iterations)
    epsilon[zero_noise_time:] = 0

    for t in range(iterations):
        X[t + 1] = X[t] * exp_ + epsilon[t]  # OU solution
        X[t + 1] = X[t + 1] if (X[t + 1] > 0.0) else 0.0  # reflecting boundary
    return X


def expected_time(kappa, x_max, C, Nspace=10000):  # pragma: no cover
    """Calculate the expected time to reach x_max starting from x_0=0.

        We assume that x_max = C sigma/ sqrt(2 kappa)
        The code solves the ODE:
            -kappa x U'(x) + 1/2 sigma^2 U''(x) = -1
        with   U'(0)=0 and U(x_max) = 0
        We use backward discretization for the first order derivative.
        Using the BC we get U(-1)=U(0) and U(N)=0.
        In matrix form, we solve:  DU = -1, with tri-diagonal matrix D.

    Args:
        kappa (float): mean reversion coefficient.
        x_max (float): maximum value for the radius.
        C (float): number of st dev distance from origin
        Nspace (int): number of discretization points.

    Returns:
        float: expected time.
    """
    x_0 = 0  # starting point of the OU process
    sigma = x_max * np.sqrt(2 * kappa) / C
    x, dx = np.linspace(x_0, x_max, Nspace, retstep=True)  # space discretization

    U = np.zeros(Nspace)  # initialization of the solution
    constant_term = -np.ones(Nspace - 1)  # -1

    sig2_dx = (sigma * sigma) / (dx * dx)
    # Construction of the tri-diagonal matrix D
    # a, lower diagonal
    a = kappa * x[:-1] / dx + 0.5 * sig2_dx
    aa = a[1:]
    # b, main diagonal
    b = -kappa * x[:-1] / dx - sig2_dx
    b[0] = -0.5 * sig2_dx  # from BC at x0
    # c, upper diagonal
    c = 0.5 * sig2_dx * np.ones_like(a)
    cc = c[:-1]

    D = sparse.diags([aa, b, cc], [-1, 0, 1], shape=(Nspace - 1, Nspace - 1)).tocsc()

    U[:-1] = spsolve(D, constant_term)
    return U[0]


def compute_OU_params(time, x_max, c):
    """Zero finder function to compute the value of kappa and sigma.

    Args:
        time (float): simulation time.
        x_max (float): maximum value for the radius.
        c (float): number of st dev distance from origin

    Returns:
        tuple: (kappa, sigma)
    """

    def obj_fun(kappa):
        """Objective function. We want to find the zero.

        Args:
            kappa (float): mean reversion coefficient.

        Returns:
            float: expected time.
        """
        return time - expected_time(kappa, x_max, c)

    x, r = scpo.brentq(obj_fun, a=1e-2, b=50, xtol=1e-8, rtol=1e-4, full_output=True)
    if r.converged:
        kappa = x
        sigma = x_max * np.sqrt(2 * kappa) / c
        return kappa, sigma

    raise ValueError
