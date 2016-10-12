#!/usr/bin/python
# -*- coding: utf-8 -*-
# Pierre Haessig — October 2016
""" Optimization of the heating of a single room

optimization objective is:

    c'.u + α ‖y-y*‖²
    with constraints u_min ≤ u ≤ u_max

where u is the heating power, y is the temperature and y* the setpoint.
Therefore, it is a compromise between the energy cost and a deviation
from tracking the reference.

"""

from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt

import dmpc

# Create the dynamics

from therm_data import dt, C, R, P_max
print('Timestep dt: {} h'.format(dt))
print('Capacity C: {} kWh/K'.format(C))
print('Resistance R: {} °C/kW'.format(R))
print('P_max: {} kWh'.format(P_max))


dyn = dmpc.dyn_from_thermal(R, C, dt, 'single room')

# Create the MPC controller
th = 2 # hours
nh = int(th/dt)
ctrl = dmpc.MPC(dyn, nh, u_min=0, u_max=P_max, u_cost=1, track_weight=100)


# Create the simulation context, e.g. forecast of perturbation and set points
from therm_data import T0, T_out, occupancy, T_abs, T_pres
print('Init temp: {} °C'.format(T0))
print('Outside temp: {} °C'.format(T_out))
print('Temp setpoints: {} °C (absent), {} °C (present)'.format(T_abs, T_pres))

n_sim = int(24/dt)

t_fcast = np.arange(n_sim+nh)*dt

occ = occupancy(t_fcast)
T_set = np.zeros(n_sim+nh) + T_abs # °C
T_set[occ] = T_pres

# Forecast of setpoint temperature: perfect
T_set_fcast = dmpc.Oracle(T_set)

# Forecast of outside temperature: perfect
T_out_fcast = dmpc.ConstOracle(T_out) # °C


ctrl.set_oracles(T_set_fcast, T_out_fcast)


# Run MPC simulation:
T0 = np.atleast_2d(T0)

P, T_out, _, T, ys = ctrl.closed_loop_sim(T0, n_sim)

# Plotting
def plot_heat_traj(t, T, T_out, T_min, P):
    'plot of trajectories from a heating simulation'
    with plt.style.context(['seaborn-deep', 'whitegrid.mplstyle']):
        fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(6,4))
        
        ax1.plot(t, T_min, label='T_set')
        ax1.plot(t, T, label='T')
        
        ax1.legend(loc='upper left')

        ax2.plot(t, P, 'r')

        ax1.set(
            ylabel='Temp (°C)',
            ylim=(T_abs-0.5, T_pres+0.5)
        )
        ax2.set(
            xlabel='t (h)',
            ylabel='P (kW)',
            ylim=(P_max*-.05, P_max*1.05)
        )
        fig.tight_layout()
    
    return fig, (ax1, ax2)

t = np.arange(n_sim)*dt
fig, (ax1, ax2) = plot_heat_traj(t, T, T_out, ys, P)
ax1.set_title('Heating MPC with (quadratic error + energy) cost')

fig.tight_layout()
fig.savefig('heating_single.png', dpi=150)
plt.show()

