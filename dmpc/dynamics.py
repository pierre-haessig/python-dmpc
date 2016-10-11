#!/usr/bin/python
# -*- coding: utf-8 -*-
# Pierre Haessig — October 2016
""" Small toolbox to create and represent
the dynamics of a dynamical system.

Classes

* LinDyn(A, Bu, Bp, C, name=None)

Functions:

* dyn_from_thermal(r_th, c_th, dt, name=None)
"""

from __future__ import division, print_function, unicode_literals
import numpy as np

from dmpc import pred_mat

class LinDyn(object):
    "Dynamics of a discrete time linear system (LTI)"
    def __init__(self, A, Bu, Bp, C, name=None):
        '''
        A, Bu, Bp, C: 2d arrays representing dynamics
        
          x⁺ = A.x + Bᵤ.u + Bₚ.p
          y  = C.x
        
        A:  (nx,nx) array
        Bu: (nx,nu) array
        Bu: (nx,np) array
        C:  (ny,nx) array
        '''
        self.name = name
        
        assert A.shape[0] == A.shape[1]
        self.nx = A.shape[0]
        
        assert Bu.shape[0] == A.shape[0]
        self.nu = Bu.shape[1]
        
        assert Bp.shape[0] == A.shape[0]
        self.np = Bp.shape[1]
        
        assert C.shape[1] == A.shape[1]
        self.ny = C.shape[0]
        
        self.A = A
        self.Bu = Bu
        self.Bp = Bp
        self.C = C

    def get_AB(self):
        'stacked A,Bu,Bp matrices'
        return np.hstack([self.A, self.Bu, self.Bp])
    
    def x_next(self, x, u, p):
        '''next state x⁺ = A.x + Bᵤ.u + Bₚ.p'''
        return self.A.dot(x) + self.Bu.dot(u) + self.Bp.dot(p)
    
    def y(self, x):
        '''output y=C.x'''
        return self.C.dot(x)
    
    def pred_mat(self, n):
        '''Constructs prediction matrices F, Hu, Hp for horizon n
        such that y = Fx + Hᵤ.u + Hₚ.p
        
        see also `pred_mat` function
        '''
        return pred_mat(n, self.A, self.C, self.Bu, self.Bp)
    
    def __str__(self):
        name = '' if self.name is None else "'{}'".format(self.name)
        return 'LinDyn {name} \n  dims x:{0.nx}, u:{0.nu}, p:{0.np}, y:{0.ny}'.format(self, name=name)


def dyn_from_thermal(r_th, c_th, dt, name=None):
    r_th = np.asarray(r_th, dtype=float).ravel()
    c_th = np.asarray(c_th, dtype=float).ravel()
    assert r_th.shape == c_th.shape
    
    tau = r_th*c_th

    A = np.diag(1 -dt/tau)
    Bu = np.diag(dt/c_th)
    B_Text = np.diag(dt/tau)
    C = np.identity(len(c_th))
    
    return LinDyn(A, Bu, B_Text, C, name=name)


if __name__ == '__main__':
    dt = 0.1 # h
    dyn = dyn_from_thermal(r_th=20, c_th=0.1, dt=dt)
