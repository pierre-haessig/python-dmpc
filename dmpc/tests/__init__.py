#!/usr/bin/python
# -*- coding: utf-8 -*-
# Pierre Haessig — October 2016

""" Test suite for dmpc. Run it with `dmpc.tests.run()`
"""

def run():
    '''run the unit tests of dmpc (requires nose)'''
    import nose
    
    import os.path
    test_dir = os.path.dirname(__file__)
    print('dmpc test suite searched in "{}"'.format(test_dir))
    
    result = nose.run(argv=['', test_dir, '-v'])
    
    if result:
        print('dmpc tests ran fine. Great!')
    else:
        print('Something went wrong within dmpc test suite.')
    return result
