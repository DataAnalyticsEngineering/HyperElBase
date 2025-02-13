#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Module torch_HypEl
==================

Description:
------------
This module provides a baseline implementation for a hyperelastic strain energy.

Metadata:
---------
Author    : Felix Fritzen
Copyright : Copyright 2025, Felix Fritzen
Credits   : GAMM AG DATA, SC SimTech
License   : MIT
Version   : 0.0.0
Maintainer: Felix Fritzen
Email     : fritzen@simtech.uni-stuttgart.de
Status    : Development

Usage:
------
To use this module, import it and call the appropriate functions:

>>> import torch_HypEl as hypel

To compute the energy:
>>> model = hypel.torch_NeoHooke(E=75.e3, nu=0.3)
>>> F = hypel.RandomDeformations(n=5, amp=0.15) # create 5 random deformations
>>> W = model.forward(F) # compute strain energy
>>> P = model.autograd_stress(F)  # compute stress via autodifferentiation


Notes:
------
- The baseline implementation torch_HypEl does not contain an actual model, but  it
  serves as a definition of the interface. Check, e.g., torch_NeoHooke.
- Implementation of the reference stress via stress(self, x) for new
  models is highly recommended and can be useful to check analytic
  expressions, too (see torch_NeoHooke as an example).
"""

#%%

import torch
from math import sqrt
torch.set_default_dtype(torch.double)



""" Utility sub-routines """
def rel_err(a, a_ref, eps=1e-3):
    """ Computes relative error

    The parameter eps is used to avoid division by zero.
    Its value depends on the chosen units!
    
    Parameters
    ----------
    a : torch.tensor
        data to be compared, shape=(n, *)
    a_ref : torch.tensor
        reference data, shape=(n, *)
        

    Returns
    -------
    torch.tensor
        absolute errors, shape=(n)
    torch.tensor
        relative errors, shape=(n)
    """
    n = a.shape[0]
    err = torch.norm( (a.detach() - a_ref.detach()).reshape((n, -1)), dim=1 )
    ref = torch.norm( a_ref.detach().reshape((n, -1)), dim=1 ) + eps
    return err, err/ref

def validate_stress( HypElModel, F, eps=1e-3):
    """ Validate the stresses of a model (analytic vs. autograd)

    Parameters
    ----------
    HypElModel : torch_HypEl
        hyperelastic model with reference stress implementation
    F : torch.tensor
        deformation gradients, shape=(n, 3, 3)
    eps : float, optional
        tolerance to avoid 1/div0 for relative errors , by default 1e-3

    Returns
    -------
    torch.tensor
        absolute errors
    torch.tensor
        relative errors
    """
    N = F.shape[0]
    F.requires_grad=True
    # W = HypElModel.forward(F)
    # P = torch.autograd.grad(W, F, torch.ones(N), retain_graph=True, create_graph=True,allow_unused=True)[0]
    P = HypElModel.autograd_stress(F)
    P_analytic = HypElModel.stress(F)

    err_abs, err_rel = rel_err(P, P_analytic, eps=1e-3)
    
    for i in range(F.shape[0]):
        print(f"rel. error {i:3d}: {100.*err_rel[i]:6.2f} %")
    return err_abs, err_rel


def RandomDeformations(n, amp=0.15):
    F = torch.zeros((n, 3, 3), dtype=torch.double)
    F = torch.eye(3).view((1, 3, 3)).repeat(n, 1, 1)
    for i in range(n):
        F[i] += (2*torch.rand(size=(3,3))-1)*amp

    J = torch.det(F)
    assert( torch.all( J>0.) ), \
        f"Generated random tensors indicate self-penetration (J<0)."

    return F

def demo_HypEl(HypElModel, n=10, amp=0.1):
    """ Demonstrates any torch_HypEl model

    An implementation of the stress via analytic expressions
    is required.

    Parameters
    ----------
    HypElModel : torch_HypEl
        hyperelastic model
    n : int
        number of samples, default is 10.
    amp : float
        amplitude of the deformation randomness, default is 0.1.
    """
    torch.set_default_dtype(torch.double)
    F = RandomDeformations(n, amp=amp)
    err_abs, err_rel = validate_stress(HypElModel, F)


def demo_NeoHooke(n=10, amp=0.1):
    """ Demonstrates the use of the NeoHooke material

    Parameters
    ----------
    n : int
        number of samples, default is 10.
    amp : float
        amplitude of the deformation randomness, default is 0.1.
    """
    mat = torch_NeoHooke(E=75.e3, nu=0.3)
    demo_HypEl(mat, n=n, amp=amp)

def demo_MooneyRivlin(n=10, amp=0.1):
    """ Demonstrates the use of the Mooney Rivlin material

    Parameters
    ----------
    n : int
        number of samples, default is 10.
    amp : float
        amplitude of the deformation randomness, default is 0.1.
    """
    mat = torch_MooneyRivlin(E=75e3, nu=0.3, C01=5.e3)
    demo_HypEl(mat, n=n, amp=amp)



class torch_HypEl (torch.nn.Module):
    """ Prototype hyperelastic model class for pytorch

    This class presents the main interface for the hyperelastic models.
    The implementation is a dummy one, i.e., the class needs to be
    specialized.
    """
    def __init__(self):
        pass
    def forward(self, x):
        """ Computes the strain energy.

        Parameters
        ----------
        x : torch.tensor
            the deformation gradients, shape (n, 3, 3) expected, dtype=torch.double

        Returns
        -------
        torch.tensor
            the strain energy for each deformation gradient
        """
        assert( x.shape[-2:] == (3,3)), \
            f"Expecting input of shape (*, 3, 3) in finite strain material models, but x.shape={x.shape}. Aborting."

        return torch.zeros(x.shape[0], dtype=torch.double)
    def stress(self, x):
        """ Computes the stresses directly.

        Parameters
        ----------
        x : torch.tensor
            the deformation gradients, shape (n, 3, 3) expected, dtype=torch.double

        Returns
        -------
        torch.tensor
            the 1st Piola Kirchhoff stress for each deformation gradient
        """
        assert( x.shape[-2:] == (3,3)), \
            f"Expecting input of shape (*, 3, 3) in finite strain material models, but x.shape={x.shape}. Aborting."
        return torch.zeros_like(x)
    
    def autograd_stress(self, x):
        """ Computes the stresses via automatic differentiation.

        Parameters
        ----------
        x : torch.tensor
            the deformation gradients, shape (n, 3, 3) expected, dtype=torch.double

        Returns
        -------
        torch.tensor
            the 1st Piola Kirchhoff stress for each deformation gradient
        """
        assert( x.shape[-2:] == (3,3)), \
            f"Expecting input of shape (*, 3, 3) in finite strain material models, but x.shape={x.shape}. Aborting."
        N = x.shape[0]
        W = self.forward(x)
        P = torch.autograd.grad(W, x, torch.ones(N), retain_graph=True, create_graph=True,allow_unused=True)[0]
        return P


class torch_NeoHooke(torch_HypEl):
    def __init__(self, E=75.e3, nu=0.3):
        """ Hyperelastic compressible Neo Hooke material

        Parameters
        ----------
        E : float, optional
            Young's modulus, by default 75.e3
        nu : float, optional
            Poisson's ratio, by default 0.3
        """
        assert(E>0), \
            f"Received E={E} MPa but expected E>0. Check input data for consistency."

        self.E = E
        self.nu = nu
        self.G = self.E/(2.*(1.+self.nu))
        self.K = self.E/(3.*(1.-2*self.nu))
        self.lam = self.K - 2./3.*self.G

    def forward(self, x):
        """ Computes the (compressible) Neo Hooke strain energy.

        Parameters
        ----------
        x : torch.tensor
            the deformation gradients, shape (n, 3, 3) expected, dtype=torch.double

        Returns
        -------
        torch.tensor
            the strain energy for each deformation gradient
        """
        assert( x.shape[-2:] == (3,3)), \
            f"Expecting input of shape (*, 3, 3) in finite strain material models, but x.shape={x.shape}. Aborting."
        # compute right Cauchy Green tensor for each deformation
        C = torch.bmm(x.transpose(1, 2), x)
        # determinant of F
        J = torch.linalg.det(x)
        # principal invariants of C
        I1 = torch.einsum("ijj->i", C)
        I2 = 0.5*(I1**2 - torch.norm(C, dim=(1, 2))**2)
        W = self.G/2 * (I1*J**(-2/3) - 3) + self.lam/2*(J-1)**2
        return W

    def stress(self, x):
        """ Analytical stress for reference.

        Parameters
        ----------
        x : torch.tensor
            the deformation gradients, shape (n, 3, 3) expected, dtype=torch.double
        
        Returns
        -------
        torch.tensor
            the 1st Piola Kirchoff stress tensors
        
        """
        assert( x.shape[-2:] == (3,3)), \
            f"Expecting input of shape (*, 3, 3) in finite strain material models, but x.shape={x.shape}. Aborting."
        C = torch.bmm(x.transpose(1, 2), x)
        J = torch.linalg.det(x)
        I1 = torch.einsum("ijj->i", C)
        I2 = 0.5*(I1**2 - torch.norm(C, dim=(1, 2))**2)
        FinvT = torch.inverse(x).transpose(1, 2)
        P = self.G*(J**(-2/3))[:,None,None]*(x - I1[:, None, None]/3. * FinvT) \
              + self.lam * ((J-1.)*J)[:, None, None] * FinvT
        return P

class torch_MooneyRivlin(torch_HypEl):
    def __init__(self, E=75.e3, nu=0.3, C01=0.):
        """ Hyperelastic compressible Mooney Rivlin material

        Parameters
        ----------
        E : float, optional
            Young's modulus, by default 75.e3
        nu : float, optional
            Poisson's ratio, by default 0.3
        C01 : float, optional
            Stiffness for the second incompressible invariant, by default 0.0
        """
        assert(E>0), \
            f"Received E={E} MPa but expected E>0. Check input data for consistency."
        assert(C01>=0.), \
            f"Received C01={C01} MPa but expected E>0. Check input data for consistency."

        self.E = E
        self.nu = nu
        self.G = self.E/(2.*(1.+self.nu))
        self.K = self.E/(3.*(1.-2*self.nu))
        self.C01 = C01
        self.C10 = self.G/2. - self.C01
        self.lam = self.K - 2./3.*self.G

    def forward(self, x):
        """ Computes the (compressible) Neo Hooke strain energy.

        Parameters
        ----------
        x : torch.tensor
            the deformation gradients, shape (n, 3, 3) expected, dtype=torch.double

        Returns
        -------
        torch.tensor
            the strain energy for each deformation gradient
        """
        assert( x.shape[-2:] == (3,3)), \
            f"Expecting input of shape (*, 3, 3) in finite strain material models, but x.shape={x.shape}. Aborting."
        # compute right Cauchy Green tensor for each deformation
        C = torch.bmm(x.transpose(1, 2), x)
        # determinant of F
        J = torch.linalg.det(x)
        # principal invariants of C
        I1 = torch.einsum("ijj->i", C)
        I2 = 0.5*(I1**2 - torch.norm(C, dim=(1, 2))**2)
        W = self.C10 * (I1*J**(-2/3) - 3) \
            + self.C01 * (I2*J**(-4/3) - 3) \
            + self.lam/2*(J-1)**2
        return W

    def stress(self, x):
        """ Analytical stress for reference.

        Parameters
        ----------
        x : torch.tensor
            the deformation gradients, shape (n, 3, 3) expected, dtype=torch.double
        
        Returns
        -------
        torch.tensor
            the 1st Piola Kirchoff stress tensors
        
        """
        assert( x.shape[-2:] == (3,3)), \
            f"Expecting input of shape (*, 3, 3) in finite strain material models, but x.shape={x.shape}. Aborting."
        C = torch.bmm(x.transpose(1, 2), x)
        J = torch.linalg.det(x)
        I1 = torch.einsum("ijj->i", C)
        I1bar = I1*(J**(-2/3))
        I2 = 0.5*(I1**2 - torch.norm(C, dim=(1, 2))**2)
        I2bar = I2*(J**(-4/3))
        FinvT = torch.inverse(x).transpose(1, 2)
        P = self.C10*(J**(-2/3))[:,None,None]*(x - I1[:, None, None]/3. * FinvT) \
              + self.C01 * ( -4./3. * I2bar[:, None, None] * FinvT  \
                            + (J**(-4/3))[:, None, None]*2.0*( I1[:, None, None] * x - torch.bmm(x, C)))  \
            + self.lam * ((J-1.)*J)[:, None, None] * FinvT
        return P

demo_MooneyRivlin()
# %%
