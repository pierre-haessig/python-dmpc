#!/usr/bin/python
# -*- coding: utf-8 -*-
# Pierre Haessig — July 2016
""" module for the distributed aspect of MPC
"""

from __future__ import division, print_function, unicode_literals
import numpy as np

__all__ = ['Coordinator']


class Coordinator(object):
    '''Coordinator for a Distributed MPC simulation
    
    The coordinator implements an iterative algorithm (Uzawa's)
    to enforce a global constraint on the sum of the input of
    each subsystem `i`:
    
        Σ_i u_i(k) ≤ u_sum_max, at all times k
    '''
    def __init__(self, subsystems, u_sum_max, u_sum_tol):
        '''
        Parameters
        ----------
        subsystems : list of MPC controllers
            each of the MPC controllers must support updating the 
        u_sum_max : scalar float
        u_sum_tol : scalar float
        '''
        self.subsystems = subsystems
        self.nh = subsystems[0].nh
        assert all([subsys.nh == self.nh for subsys in subsystem]), \
            "all subsystems should have same MPC horizon"
        # TODO: assert all subsystems are SISO: n_u==1, n_y == 1
        self.set_u_sum_max(self, u_sum_max)
        self.set_u_sum_tol(self, u_sum_tol)
    
    def set_u_sum_max(self, u_sum_max):
        '''sets the global constraint on the sum of inputs of the subsystems.
        
        Σ_i u_i(k) ≤ u_sum_max, at all times k
        
        Parameters
        ----------
        u_sum_max : scalar float
        '''
        self.u_sum_max = u_sum_max
    
    def set_u_sum_tol(self, u_sum_tol):
        '''sets the tolerance of the global constraint.
        
        Uzawa iteration stop when:
        
        |Σ_i u_i(k) - u_sum_max| ≤ u_sum_tol
        
        Parameters
        ----------
        u_sum_tol : scalar float
        '''
        self.u_sum_tol = u_sum_tol
    
    def closed_loop_sim(self, x0, n_sim, dyns=None):
        '''Simulates the D-MPC in closed loop
        
        Returns u, p, x, y, ys, l
        '''
        if dyns is None:
            dyns = [c.dyn for c in self.subsystems]
        
        n_sys = len(self.subsystems)
        assert len(dyns) == n_sys
        
        nh = self.nh
        
        n_x_list = [d.A.shape[0] for d in dyns]
        n_x = sum(n_x_list)
        n_y_list = [d.C.shape[0] for d in dyns]
        n_y = sum(n_y_list)
        n_u_list = [d.Bu.shape[1] for d in dyns]
        n_u = sum(n_u_list)
        n_p_list = [d.Bp.shape[1] for d in dyns]
        n_p = sum(n_p_list)
        assert x0.shape == (n_x,1) or x0.shape == (n_x,)
        
        u = np.zeros((n_sim, n_u))
        p = np.zeros((n_sim, n_p))
        x = np.zeros((n_sim, n_x))
        y = np.zeros((n_sim, n_y))
        ys = np.zeros((n_sim, n_y))
        l = np.zeros((n_sim, 1))
        
        x[0] = x0.reshape(-1)
        
        for k in range(n_sim): # time loop
            # collect forecasts: (assumed independent of multiplier)
            p_hor_list = [c.p_fcast.pred(k, nh) for c in self.subsystems]
            # TODO: fix the semantics. Output forecast should be k+1:k+1:nh | k
            ys_hor_list = [c.ys_fcast.pred(k+1, nh) for c in self.subsystems]
            
            x_k = x[k]
            # collect outputs and output references
            for ks, c in enumerate(self.subsystems): # iteration over subsystems
                y[k, ks] = dyns[ks].y(x_k[ks])
                ys[k, ks] = c.ys_fcast.real(k).reshape(-1)
                # init each mpc
                c.set_xyp(x_k[ks], ys_hor_list[ks], p_hor_list[ks])
            # end for each subsystem
            
            # init multiplier:
            l_hor = np.zeros((nh, 1))
            u_hor_list = np.zeros((nh, n_u))
            
            # Negociation with subsystems
            for ku in range(ku_max): # Uzawa iteration
                for ks, c in enumerate(self.subsystems): # iteration over subsystems
                    c.set_u_mult(l_hor)
                    u_hor_list[:,ks] = c.solve_u_opt()
                # end for each subsystem
            
                # Check constraint and update multiplier
                u_sum = u_hor_list.sum(axis=1)
                u_sum_excess = u_sum - u_sum_max
                
                if np.all(np.abs(u_sum_excess)-u_sum_tol):
                    # Uzawa converged
                    break
                
                l_hor += step*(u_sum_excess)
            # end Uzawa iteration
            
            l[k] = l_hor[0]
            
            # collect input, perturbation and next states
            for ks, c in enumerate(self.subsystems): # iteration over subsystems
                #y[k, ks] = dyns[ks].y(x_k[ks])
                #ys[k, ks] = c.ys_fcast.real(k).reshape(-1)
                u_hor = u_hor_list[:,ks]
                u[k, ks] = u_hor.reshape((nh, 1))[0]
                p[k, ks] = c.p_fcast.real(k).reshape(-1)
                
                if k+1<n_sim:
                    x[k+1] = dyns[ks].x_next(x_k[ks], u[k, ks], p[k, ks])
            # end for each subsystem
        # end for each instant k
        
        return u, p, x, y, ys
