#!/usr/bin/python
# -*- coding: utf-8 -*-
# Pierre Haessig — July 2016
""" A toolbox for Model Predictive Control (MPC) and Distributed MPC.
"""

from __future__ import division, print_function, unicode_literals
import numpy as np

import warnings
import cvxopt

from .mat_help import block_toeplitz

cvxopt.solvers.options['show_progress'] = False


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


def pred_mat(n, A, C, *B_list):
    '''
    Constructs prediction matrices F, H for horizon n
    such that y = Fx + H.u
    
    Any number of B matrices can be given, which will return an equal
    number of H matrices
    
    TODO: implement case C≠[1]
    
    Examples
    --------
    >>> F, H = pred_mat(3, np.atleast_2d(0.9), np.atleast_2d(1), np.atleast_2d(0.2))
    >>> F
    array([[ 0.9  ],
           [ 0.81 ],
           [ 0.729]])
    >>> H
    array([[ 0.2  ,  0.   ,  0.   ],
           [ 0.18 ,  0.2  ,  0.   ],
           [ 0.162,  0.18 ,  0.2  ]])
    with 2 B matrices:
    >>> F, Hu, Hp = pred_mat(3, np.atleast_2d(0.9), np.atleast_2d(1), np.atleast_2d(0.2), np.atleast_2d(0.5))
    >>> Hp
    array([[ 0.5  ,  0.   ,  0.   ],
           [ 0.45 ,  0.5  ,  0.   ],
           [ 0.405,  0.45 ,  0.5  ]])
    '''
    # [A^i] for i=1:n
    A_pow = [A]
    for i in range(n-1):
        A_pow.append(A_pow[-1].dot(A))
    F = np.vstack(A_pow)

    H_list = []
    for B in B_list:
        # [A^i.Bu] for i=0:n-1
        AB_pow = [B]
        AB_pow.extend([Ai.dot(B) for Ai in A_pow[:-1]])
        
        H = block_toeplitz(AB_pow, np.zeros(n))
        H_list.append(H)
    
    return (F,) + tuple(H_list)



class MPC(object):
    '''MPC controller
    
    with cost function c'.u + α ‖y-y*‖²
    constraints u_min ≤ u ≤ u_max
    '''
    def __init__(self, dyn, nh, u_min, u_max, u_cost, track_weight):
        self.dyn = dyn
        self.nh = nh
        self.u_min = u_min
        self.u_max = u_max
        self._u_cost = u_cost
        self._u_mult = None
        self.track_weight = track_weight
        
        # Prediction matrices
        F, Hu, Hp = dyn.pred_mat(nh)
        self.F = F
        self.Hu = Hu
        self.Hp = Hp
        
        # Precompute quadprog matrices which are independent of time
        self._update_P()
        self._update_Gh()
        
        self.ys_fcast = None
        self.p_fcast = None
    
    def set_oracles(self, ys_fcast, p_fcast):
        '''sets the input forecast generators (oracles) for ys and p
        
        ys_fcast : Oracle for ys, the output setpoint.
        
        p_fcast : Oracle for p, the input perturbation.
        '''
        assert hasattr(ys_fcast, 'pred') and hasattr(ys_fcast, 'real')
        self.ys_fcast = ys_fcast
        
        assert hasattr(p_fcast, 'pred') and hasattr(p_fcast, 'real')
        self.p_fcast = p_fcast
    
    def get_pred_mat(self):
        return self.F, self.Hu, self.Hp
    
    def pred_output(self, x0, u_hor, p_hor, reshape_2d=False):
        '''predicted future outputs of the system
        
        given
        * initial state `x0` and
        * predicted inputs `u_hor` and `p_hor` over the horizon
        
        if `reshape_2d` is True, returns a (nh,ny) array instead of the flat
        (nh*ny,1) column vector (default).
        
        formula y = Fx + Hu.u + Hp.p
        '''
        F = self.F
        Hu = self.Hu
        Hp = self.Hp
        y_hor = np.dot(F, x0) + np.dot(Hu,u_hor) + np.dot(Hp, p_hor)
        
        if reshape_2d:
            y_hor = y_hor.reshape((self.nh, self.dyn.ny))
        
        return y_hor
    
    def _update_P(self):
        track_weight = self.track_weight
        Hu = self.Hu
        self.P = 2*track_weight * (Hu.T).dot(Hu)
    
    def _update_q(self, u_cost, x0=None, ys_hor=None, p_hor=None):
        if x0 is not None:
            track_weight = self.track_weight
            F = self.F
            Hu = self.Hu
            Hp = self.Hp
            F_x = F.dot(x0).reshape(-1)
            Hp_p = Hp.dot(p_hor).reshape(-1)
            ys_hor = ys_hor.reshape(-1)
            self._q0 = 2*track_weight*(Hu.T).dot(F_x + Hp_p - ys_hor)
        
        q = u_cost + self._q0
        assert q.shape == (q.size, 1) or q.ndim == 1
        self.q = q
    
#    def _update_j0(self, x0, ys_hor, p_hor):
#        j0 = ys_hor.T.dot(ys_hor - 2*(F_x + Hp_p)) + \
#             F_x.T.dot(F_x) + Hp_p.T.dot(Hp_p) + 2*F_x.T.dot(Hp_p)
#        self.j0 = j0*track_weight
    
    def _update_Gh(self):
        F = self.F
        u_min = self.u_min
        u_max = self.u_max

        n = F.shape[0]
        In = np.identity(n)
        self.G = np.vstack((-In, In))
        ones_n = np.ones(n)
        self.h = np.hstack((-u_min*ones_n, u_max*ones_n))
    
#    @staticmethod
#    def qp_mat(F, Hu, Hp, x0, u_cost, track_weight):
#        '''
#        Construct quadprog matrices P, q, j0, G, h
#        
#        corresponding to ``cvxopt.solvers.qp`` notation, with constant:
#        
#            minimize    (1/2) x'.P.x + q'.x + j0
#            subject to  G.x <= h
#        '''
#        # P
#        P = 2*track_weight * (Hu.T).dot(Hu)
#        # q
#        F_x = F.dot(x0)
#        Hp_p = Hp.dot(p_hor)
#        q = u_cost + 2*track_weight*(Hu.T).dot(F_x + Hp_p - ys_hor)
#        
#        #j0
#        j0 = ys_hor.T.dot(ys_hor - 2*(F_x + Hp_p)) + \
#             F_x.T.dot(F_x) + Hp_p.T.dot(Hp_p) + 2*F_x.T.dot(Hp_p)
#        j0 = j0*track_weight
#        # G
#        n = F.shape[0]
#        In = np.identity(n)
#        G = np.vstack((-In, In))
#        # h
#        h = np.hstack((np.zeros(n), np.ones(n) * u_max))
#        
#        return P, q, j0, G, h
    
    def set_xyp(self, x0, ys_hor, p_hor):
        '''sets the state measurement x0, and forecasts on the horizon
        set point ys_hor and perturbation p_hor
        '''
        if self._u_mult is None:
            u_cost = self._u_cost
        else:
            u_cost = self._u_cost + self._u_mult
        self._update_q(u_cost, x0, ys_hor, p_hor)
        #self._update_j0(x0, ys_hor, p_hor)
    
    def set_u_mult(self, u_mult):
        '''recompute quadprog matrices with new Lagrange multiplier for u
        
        (u_mult gets added to u_cost)
        '''
        self._u_mult = u_mult
        u_cost = self._u_cost + self._u_mult
        self._update_q(u_cost)
    
    def solve_u_opt(self, full_output=False, reshape_2d=False):
        '''solves for best sequence of inputs `u` over horizon `self.nh`
        by solving the minization problem.
        
        Solver used: `cvxopt.solvers.qp`
        
        Returns
        -------
        u_opt : sequence (column vector).
        If `full_output`, returns `u_opt` and also output dict of qp solver.
        If `reshape_2d` is True, returns a (nu,ny) array instead of the flat
        (nu*ny,1) column vector (default).
        
        '''
        matrix = cvxopt.matrix
        cxP = matrix(self.P)
        cxq = matrix(self.q)
        cxG = matrix(self.G)
        cxh = matrix(self.h)
        sol = cvxopt.solvers.qp(cxP, cxq, cxG, cxh)
        if not sol['status'] == 'optimal':
            msg = "cvxopt.solvers.qp didn't converged (return status '{}')".format(sol['status'])
            warnings.warn(msg)
        u_opt = np.array(sol['x'])
        
        if reshape_2d:
            u_opt = u_opt.reshape((self.nh, self.dyn.nu))
        
        if full_output:
            return u_opt, sol
        return u_opt

    def closed_loop_sim(self, x0, n_sim, dyn=None):
        '''Simulates the MPC in closed loop
        
        Dynamics `dyn` can be the same as `self.dyn`, or different
        (e.g. to investigate the effect modeling errors).
        If `None` (default), falls back to `self.dyn`.
        
        Returns u, p, x, y, ys
        '''
        if dyn is None:
            dyn = self.dyn
        
        A = dyn.A
        Bu = dyn.Bu
        Bp = dyn.Bp
        C = dyn.C
        
        nh = self.nh
        
        n_x = A.shape[0]
        n_y = C.shape[0]
        n_u = Bu.shape[1]
        n_p = Bp.shape[1]
        assert x0.shape == (n_x,1) or x0.shape == (n_x,)
        
        u = np.zeros((n_sim, n_u))
        p = np.zeros((n_sim, n_p))
        x = np.zeros((n_sim, n_x))
        y = np.zeros((n_sim, n_y))
        ys = np.zeros((n_sim, n_y))
        x[0] = x0.reshape(-1)
        
        for k in range(n_sim):
            p_hor = self.p_fcast.pred(k, nh)
            
            # TODO: fix the semantics. Output forecast should be k+1:k+1:nh | k
            ys_hor = self.ys_fcast.pred(k+1, nh)
            
            x_k = x[k]
            y[k] = dyn.y(x_k)
            ys[k] = self.ys_fcast.real(k).reshape(-1)
            
            # mpc:
            self.set_xyp(x_k, ys_hor, p_hor)
            u_hor = self.solve_u_opt()
            
            u[k] = u_hor.reshape((nh, n_u))[0]
            p[k] = self.p_fcast.real(k).reshape(-1)
            
            if k+1<n_sim:
                x[k+1] = dyn.x_next(x_k, u[k], p[k])
        # end for each instant k
        
        return u, p, x, y, ys


class Oracle(object):
    '''Perfect predictor of signal y
    '''
    def __init__(self, y):
        self.y = y
    
    def pred(self, k, n):
        '''n future values of y, starting from instant k: 
        
            y[k:k+n]
        '''
        return self.y[k:k+n].reshape((-1,1))
    
    def real(self, k):
        '''real value (realization) of y at instant k'''
        return self.y[k].reshape((-1,1))


class ConstOracle(Oracle):
    '''Perfect predictor of constant signal yc
    '''
    def __init__(self, yc):
        self.yc = np.array(yc).ravel()
        self.ny = len(self.yc)
    
    def pred(self, k, nh, flat=True):
        '''nh future values of y from instant k: y[k:k+nh]
        '''
        z_nh = np.zeros((nh, 1))
        y_hor_2d = self.yc + z_nh
        if flat:
            y_hor = y_hor_2d.reshape((-1, 1))
        else:
            y_hor = y_hor_2d
        return y_hor
    
    def real(self, k):
        '''real value (realization) of y at instant k'''
        return self.yc.reshape((-1,1))


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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    dt = 0.1 # h
    dyn = dyn_from_thermal(r_th=20, c_th=0.1, dt=dt)
    
    nh = int(2/dt)
    ctrl = MPC(dyn, nh, u_min=0, u_max=1.5, u_cost=1, track_weight=100)
    
    
    T0 = np.atleast_2d(20)# °C
    
    # check MPC output at one time step:
    zn = np.zeros(nh)[:,None]
    T_ext_hor = 2 + zn # °C
    Ts_hor = 18 + zn # °C
    Ts_hor[nh//2:] = 22 # °C
    
    ctrl.set_xyp(T0, Ts_hor, T_ext_hor)
    u_opt = ctrl.solve_u_opt()
    
    T_pred = ctrl.pred_output(T0, u_opt, T_ext_hor)
    
    t_hor = np.arange(nh)*dt
    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
    
    ax1.plot(t_hor+dt, Ts_hor, 'k:', label='set point')
    ax1.plot(t_hor+dt, T_pred, 'g+-', label='MPC prediction')
    ax1.plot(0, T0, 'gD')
    
    ax1.set(
        title = 'MPC sequence at one step (dt={:.1f}h, nh={})'.format(dt,nh),
        ylabel = u'temperature (°C)'
        )
    ax1.legend()
        
    ax2.plot(t_hor, u_opt, 'r+-')
    ax2.set(
        xlabel = 'time (h)',
        ylabel = u'heating (kW)'
        )
    
    ### Closed loop simulation
    n_sim = int(6/dt)
    
    Ts_fcast_arr = 18 + np.zeros(n_sim+nh) # °C
    Ts_fcast_arr[n_sim//2:] = 22 # °C
    Ts_fcast = Oracle(Ts_fcast_arr)
    
    T_ext_fcast = ConstOracle(2) # °C
    
    ctrl.set_oracles(Ts_fcast, T_ext_fcast)
    
    u, p, x, y, ys = ctrl.closed_loop_sim(T0, n_sim)
    
    t_sim = np.arange(n_sim)*dt
    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
    ax1.plot(t_sim, ys, 'k:', label='T set point')
    ax1.plot(t_sim, y, 'g+-', label='T_sim')
    ax1.plot(0, T0, 'gD')
    
    ax1.set(
        title = 'MPC in closed loop (dt={:.1f}h, nh={}, n_sim={})'.format(dt,nh, n_sim),
        ylabel = u'temperature (°C)'
        )
    ax1.legend()
    
    ax2.plot(t_sim, u, 'r+-')
    ax2.set(
        xlabel = 'time (h)',
        ylabel = u'heating (kW)'
        )
    
    plt.show()
