#!/usr/bin/python
# -*- coding: utf-8 -*-
# Pierre Haessig — May 2016
""" Data for the thermal control problem

choice of units: hours, kW, Celsius degrees
"""
from __future__ import unicode_literals, print_function, division

import numpy as np

# time step:
dt = 0.1 # h

# thermal capacity
C = 1 # kWh/K

# thermal resistance with outside:
R = 20 # °C/kW

# Maximum heating power:
P_max = 3 # kW

# Initial temperature
T0 = 19 # °C

# Outdoor temperature
T_out = 2 # °C

# Temperature set points:
# absent
T_abs = 18 # °C
# present
T_pres = 22 # °C


def occupancy(t, t_switch = ((6.5, 8), (18, 22))):
    '''boolean occupancy vector for each instant in vector `t` (in hours)
    
    occupancy is True between switching hours `t_switch`,
    which is a list of pairs of hours (t_on, t_off).
    
    By default, occupancy is True:
    
    * in the morning 6:30 to 8:00
    * in the evening: 18:00 to 22:00
    '''
    h = t % 24
    occ = np.zeros_like(t, dtype=bool)
    for (t_on, t_off) in t_switch:
        assert t_off >= t_on
        occ |= (h>=t_on) & (h<=t_off) 
    return occ
