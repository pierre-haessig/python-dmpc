#!/usr/bin/python
# -*- coding: utf-8 -*-
# Pierre Haessig — October 2016
""" A toolbox for Model Predictive Control (MPC) and Distributed MPC.
"""

from .mpc import *
from .dynamics import *
from .distributed import *
from . import mat_help


def run_tests():
    '''run the unit tests (requires nose)'''
    import nose
    result = nose.run()
    if result:
        print('dmpc tests ran fine. Great!')
    else:
        print('Something went wrong within dmpc test suite.')
    return result
