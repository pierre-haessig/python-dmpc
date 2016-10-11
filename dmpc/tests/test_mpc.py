#!/usr/bin/python
# -*- coding: utf-8 -*-
# Pierre Haessig — July 2016
""" test the MPC toolbox

"""

from __future__ import division, print_function, unicode_literals
from nose.tools import assert_true, assert_equal
from numpy.testing import assert_allclose
import numpy as np

def assert_allclose9(a,b):
    return assert_allclose(a, b, 1e-9, 1e-9)

def assert_allclose6(a,b):
    return assert_allclose(a, b, 1e-6, 1e-6)


import mpc

def test_dyn_from_thermal():
    dyn = mpc.dyn_from_thermal(5, 1, dt=0.1)
    assert_equal(dyn.A, 0.98)
    assert_equal(dyn.Bu, 0.1)
    assert_equal(dyn.Bp, 0.02)
    assert_equal(dyn.C, 1)
    
    dyn2 = mpc.dyn_from_thermal([5,5], [1,1], dt=0.1)
    I2 = np.identity(2)
    assert_allclose9(dyn2.A, 0.98*I2)
    assert_allclose9(dyn2.Bu, 0.1*I2)
    assert_allclose9(dyn2.Bp, 0.02*I2)
    assert_allclose9(dyn2.C, I2)

def test_block_toeplitz():
    assert_allclose9(
        mpc.block_toeplitz([1,2,3], [1,4,5,6]),
        np.array([[1, 4, 5, 6],
                  [2, 1, 4, 5],
                  [3, 2, 1, 4]])
        )
    
    I2 = np.eye(2)
    assert_allclose9(
        mpc.block_toeplitz([1*I2,2*I2,3*I2], [1*I2,4*I2,5*I2,6*I2]),
        np.array([[1, 0, 4, 0, 5, 0, 6, 0],
                  [0, 1, 0, 4, 0, 5, 0, 6],
                  [2, 0, 1, 0, 4, 0, 5, 0],
                  [0, 2, 0, 1, 0, 4, 0, 5],
                  [3, 0, 2, 0, 1, 0, 4, 0],
                  [0, 3, 0, 2, 0, 1, 0, 4]])
        )
    
    assert_allclose9(
        mpc.block_toeplitz([1*I2,2*I2,3*I2], [1,4,5,6]),
        np.array([[1, 0, 4, 4, 5, 5, 6, 6],
                  [0, 1, 4, 4, 5, 5, 6, 6],
                  [2, 0, 1, 0, 4, 4, 5, 5],
                  [0, 2, 0, 1, 4, 4, 5, 5],
                  [3, 0, 2, 0, 1, 0, 4, 4],
                  [0, 3, 0, 2, 0, 1, 4, 4]])
        )

def get_dyn(c_th):
    '''creates a LinDyn of a thermal system'''
    r_th = 20
    dt = 0.2 #h
    dyn = mpc.dyn_from_thermal(r_th, c_th, dt, "thermal subsys")
    
    return dyn, dt

def test_pred_mat():
    '''test prediction matrices on a 2D thermal system'''
    dyn, dt = get_dyn(c_th = 0.02) # 0.4 h time constant

    n_hor = int(2.5/dt)
    assert n_hor == 12
    t = np.arange(1, n_hor+1)*dt
    F, Hu, Hp = mpc.pred_mat(n_hor, dyn.A, dyn.C, dyn.Bu, dyn.Bp)

    zn = np.zeros(n_hor)[:,None]
    T_ext_hor = 2 + zn # °C
    u_hor = 0 + zn # kW
    u_hor[t>1] = 1 #kW
    T0 = 20 # °C

    T_hor = np.dot(F, T0) + np.dot(Hu, u_hor) + np.dot(Hp, T_ext_hor)
    
    assert_equal(T_hor.shape, (12,1))
    
    assert_allclose9(
        T_hor,
        np.array([[ 11.        ],
                  [  6.5       ],
                  [  4.25      ],
                  [  3.125     ],
                  [  2.5625    ],
                  [ 12.28125   ], # u becomes 1: T goes up
                  [ 17.140625  ],
                  [ 19.5703125 ],
                  [ 20.78515625],
                  [ 21.39257812],
                  [ 21.69628906],
                  [ 21.84814453]])
        )
    
    # Also check the method of LinDyn
    F1, Hu1, Hp1 = dyn.pred_mat(n_hor)
    assert_true(np.all(F==F1))
    assert_true(np.all(Hu==Hu1))
    assert_true(np.all(Hp==Hp1))


class test_MPC():
    '''Tests for MPC class'''
    def __init__(self):
        dt = 0.2 #h
        r_th = 20
        c_th = 0.05 # 1 h time constant
        self.dt = dt
        self.r_th = r_th
        self.c_th = c_th
        
        # 1D system
        self.dyn1 = mpc.dyn_from_thermal(r_th, c_th, dt, "1d thermal 1hour")
        
        # 2D system
        self.dyn2 = mpc.dyn_from_thermal([r_th]*2, [c_th]*2, dt, "2d thermal 1hour")
    
    def test_pred_output(self):
        '''test MPC.pred_output method for 2d system'''
        dt = self.dt
        dyn2 = self.dyn2
        n_sys = 2
        assert n_sys == dyn2.nx
        
        n_hor = int(2.5/dt)
        assert n_hor == 12
        
        u_max = 1.2 #kW
        c2 = mpc.MPC(dyn2, n_hor, 0, u_max, 1, 100)
        
        # input predictions. needs to be flattened
        T_ext_hor = 2+np.zeros((n_hor, n_sys)) # °C
        T_ext_hor = T_ext_hor.reshape((-1, 1))

        u_hor = np.zeros((n_hor, n_sys))
        u_hor[n_hor//2:] = 1
        u_hor = u_hor.reshape((-1, 1))

        T0 = np.array([20,10]).reshape((n_sys,1))

        T_pred = c2.pred_output(T0, u_hor, T_ext_hor)
        assert_equal(T_pred.shape, (n_sys*n_hor, 1))
        
        T_pred = c2.pred_output(T0, u_hor, T_ext_hor, reshape_2d=True)
        assert_equal(T_pred.shape, (n_hor, n_sys))
        
        T_expected = np.array([
            [ 16.4       ,   8.4       ],
            [ 13.52      ,   7.12      ],
            [ 11.216     ,   6.096     ],
            [  9.3728    ,   5.2768    ],
            [  7.89824   ,   4.62144   ],
            [  6.718592  ,   4.097152  ],
            [  9.7748736 ,   7.6777216 ],
            [ 12.21989888,  10.54217728],
            [ 14.1759191 ,  12.83374182],
            [ 15.74073528,  14.66699346],
            [ 16.99258823,  16.13359477],
            [ 17.99407058,  17.30687581]])
        
        print(T_pred)
        assert_allclose9(T_pred, T_expected)

    def test_MPC_solve_u_opt(self):
        '''test of MPC.solve_u_opt() method (calls cvxopt)'''
        dt = self.dt
        dyn = self.dyn1

        n_hor = int(2.5/dt)
        assert n_hor == 12
        
        u_max = 1.2 #kW
        ctrl = mpc.MPC(dyn, n_hor, 0, u_max, 1, 100)

        zn = np.zeros(n_hor)[:,None]
        T_ext_hor = 2 + zn # °C
        
        Ts_hor = 18 + zn # °C
        Ts_hor[5:] = 22 # °C ()
        
        T0 = 20 # °C
        
        ctrl.set_xyp(T0, Ts_hor, T_ext_hor)
        
        u_opt = ctrl.solve_u_opt()
        
        assert_equal(u_opt.shape, (n_hor,1))
        
        u_expected = np.array([
            [ 0.39993755],
            [ 0.79998746], # steady state at 0.8 kW
            [ 0.7999875 ],
            [ 0.79998751],
            [ 1.12160221],
            [ 1.19999986],  # peak at u_max
            [ 1.19999099],
            [ 1.05931038],
            [ 0.9999868 ],  # steady state at 1 kW
            [ 0.99998794],
            [ 0.99998734],
            [ 0.99973748]])
        
        print((u_opt - u_expected))
        
        assert_allclose6(u_opt, u_expected)
    
    def test_MPC_solve_u_opt_2d(self):
        '''test of MPC.solve_u_opt() method for 2d system'''
        dt = self.dt
        dyn2 = self.dyn2
        n_sys = 2
        assert n_sys == dyn2.nx
        
        n_hor = int(2.5/dt)
        assert n_hor == 12
        
        u_max = 1.2 #kW
        c2 = mpc.MPC(dyn2, n_hor, 0, u_max, 1, 100)
        
        # input predictions. needs to be flattened
        T_ext_hor = 2+np.zeros((n_hor, n_sys)) # °C
        T_ext_hor = T_ext_hor.reshape((n_hor * n_sys, 1))
        
        Ts_hor = 18 + np.zeros((n_hor, n_sys)) # °C
        Ts_hor[n_hor//2:] = 22 # °C
        Ts_hor = Ts_hor.reshape((n_hor * n_sys, 1))
        
        T0 = np.array([20,10]).reshape((n_sys,1))

        c2.set_xyp(T0, Ts_hor, T_ext_hor)

        u_opt = c2.solve_u_opt()
        assert_equal(u_opt.shape, (n_hor * n_sys, 1))
        
        u_opt = c2.solve_u_opt(reshape_2d=True)
        assert_equal(u_opt.shape, (n_hor, n_sys))

        u_expected = np.array([
            [ 0.39993771,  1.19999994],
            [ 0.79998731,  1.19999975],
            [ 0.79998749,  1.19994822],
            [ 0.7999875 ,  0.83837907],
            [ 0.79998827,  0.80037647],
            [ 1.1215989 ,  1.12105229],
            [ 1.19999977,  1.19999963],
            [ 1.19999867,  1.19999681],
            [ 1.0593051 ,  1.05942779],
            [ 0.99998763,  0.99998742], # steady state at 1 kW
            [ 0.99998749,  0.9999875 ],
            [ 0.99973719,  0.99973719]])
        
        print(repr(u_opt))
        
        assert_allclose6(u_opt, u_expected)

    def test_MPC_closed_loop_sim(self):
        '''test MPC.closed_loop_sim() method on a short simulation'''
        dt = self.dt
        dyn = self.dyn1

        n_hor = int(1.5/dt)
        assert n_hor == 7
        n_sim = int(3/dt)
        assert n_sim == 15
        
        u_max = 1.2 #kW
        ctrl = mpc.MPC(dyn, n_hor, 0, u_max, 1, 100)

        Ts_fcast_arr = 18 + np.zeros(n_sim+n_hor) # °C
        Ts_fcast_arr[n_sim//2:] = 22 # °C
        Ts_fcast = mpc.Oracle(Ts_fcast_arr)
        
        T_ext_fcast = mpc.ConstOracle(2) # °C
        
        T0 = np.atleast_2d(20) # °C
        
        ctrl.set_oracles(Ts_fcast, T_ext_fcast)
        
        u_sim, p, x, T_sim, Ts = ctrl.closed_loop_sim(T0, n_sim)
        
        # check consistency of T set point with Oracle input
        assert_equal(Ts.shape, (n_sim,1))
        assert_allclose9(Ts, Ts_fcast_arr[0:n_sim, None])
        
        # Check temperature output
        assert_equal(T_sim.shape, (n_sim,1))
        
        T_expected = np.array([
            [ 20.        ], # Ts is 18°
            [ 17.99975035],
            [ 17.99974999],
            [ 17.99974999],
            [ 17.99974966],
            [ 17.99975053],
            [ 19.28619851],
            [ 20.62895623], # Ts becomes 22°
            [ 21.70300215],
            [ 21.99974997],
            [ 21.99974998],
            [ 21.99974998],
            [ 21.99974998],
            [ 21.99974998],
            [ 21.99974998]])
        
        print((T_sim - T_expected))
        print(np.hstack((T_sim, T_expected)))
        
        assert_allclose6(T_sim, T_expected)
        
        # Check MPC control sequence
        assert_equal(u_sim.shape, (n_sim,1))
        
        u_expected = np.array([
            [ 0.39993759],
            [ 0.79998743], # steady state at 0.8 kW
            [ 0.7999875 ],
            [ 0.79998742],
            [ 0.7999877 ],
            [ 1.12159952], # peak at u_max
            [ 1.19999936],
            [ 1.19995929],
            [ 1.05933706],
            [ 0.9999875 ], # steady state at 1 kW
            [ 0.9999875 ],
            [ 0.9999875 ],
            [ 0.9999875 ],
            [ 0.9999875 ],
            [ 0.9999875 ]])
            
        print((u_sim - u_expected))
        assert_allclose6(u_sim, u_expected)
    
    def test_MPC_closed_loop_sim_2d(self):
        '''test MPC.closed_loop_sim() method on a 2d system'''
        dt = self.dt
        dyn2 = self.dyn2
        n_sys = 2
        assert n_sys == dyn2.nx
        
        n_hor = int(1.5/dt)
        assert n_hor == 7
        n_sim = int(3/dt)
        assert n_sim == 15
        
        u_max = 1.2 #kW
        c2 = mpc.MPC(dyn2, n_hor, 0, u_max, 1, 100)

        Ts_fcast_arr = 18 + np.zeros((n_sim+n_hor, n_sys)) # °C
        Ts_fcast_arr[n_sim//2:] = 22 # °C
        Ts_fcast = mpc.Oracle(Ts_fcast_arr)
        
        T_ext_fcast = mpc.ConstOracle([2, 2]) # °C
        
        T0 = np.array([20,10]).reshape((n_sys,1)) # °C
        
        c2.set_oracles(Ts_fcast, T_ext_fcast)
        
        u_sim, p, x, T_sim, Ts = c2.closed_loop_sim(T0, n_sim)
        
        # check consistency of T set point with Oracle input
        assert_equal(Ts.shape, (n_sim, n_sys))
        assert_allclose9(Ts, Ts_fcast_arr[0:n_sim, :])
        
        # Check temperature output
        assert_equal(T_sim.shape, (n_sim, n_sys))
        
        T_expected = np.array([
            [ 20.        ,  10.        ], # Ts is 18°
            [ 17.99975046,  13.19999987],
            [ 17.99974999,  15.75999979],
            [ 17.99974999,  17.80799464],
            [ 17.99974966,  17.99974909],
            [ 17.99975053,  17.99975053],
            [ 19.28619851,  19.28619851],
            [ 20.62895623,  20.62895623], # Ts becomes 22°
            [ 21.70300215,  21.70300215],
            [ 21.99974997,  21.99974997],
            [ 21.99974998,  21.99974998],
            [ 21.99974998,  21.99974998],
            [ 21.99974998,  21.99974998],
            [ 21.99974998,  21.99974998],
            [ 21.99974998,  21.99974998]])
        
        print((T_sim - T_expected))
        print(np.hstack((T_sim, T_expected)))
        
        assert_allclose6(T_sim, T_expected)
        
        # Check MPC control sequence
        assert_equal(u_sim.shape, (n_sim, n_sys))
        
        u_expected = np.array([
            [ 0.39993761,  1.19999997],
            [ 0.7999874 ,  1.19999997], # u1 steady state at 0.8 kW, u2=u_max
            [ 0.7999875 ,  1.1999987 ],
            [ 0.79998742,  0.83833834],
            [ 0.7999877 ,  0.79998781], # steady state at 0.8 kW
            [ 1.12159952,  1.12159952],
            [ 1.19999936,  1.19999936], # peak at u_max
            [ 1.19995929,  1.19995929],
            [ 1.05933706,  1.05933706],
            [ 0.9999875 ,  0.9999875 ], # steady state at 1 kW
            [ 0.9999875 ,  0.9999875 ],
            [ 0.9999875 ,  0.9999875 ],
            [ 0.9999875 ,  0.9999875 ],
            [ 0.9999875 ,  0.9999875 ],
            [ 0.9999875 ,  0.9999875 ]])
        
        print((u_sim - u_expected))
        assert_allclose6(u_sim, u_expected)
