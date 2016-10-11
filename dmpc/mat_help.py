#!/usr/bin/python
# -*- coding: utf-8 -*-
# Pierre Haessig — July 2016
""" Matrix helper functions
"""

from __future__ import division, print_function, unicode_literals
import numpy as np


def block_toeplitz(c, r=None):
    '''
    Construct a block Toeplitz matrix, with blocks having the same shape
    
    Signature is compatible with ``scipy.linalg.toeplitz``
    
    Parameters
    ----------
    c : list of 2d arrays
        First column of the matrix.
        Each item of the list should have same shape (mb,nb)
    r : list of 2d arrays
        First row of the matrix. If None, ``r = conjugate(c)`` is assumed;
        in this case, if c[0] is real, the result is a Hermitian matrix.
        r[0] is ignored; the first row of the returned matrix is
        made of blocks ``[c[0], r[1:]]``.
    
    c and r can also be lists of scalars; if so they will be broadcasted
    to the fill the blocks
    
    Returns
    -------
    A : (len(c)*mb, len(r)*nb) ndarray
        The block Toeplitz matrix.
    
    Examples
    --------
    Compatible with ``scipy.linalg.toeplitz`` (but less optimized):
    >>> block_toeplitz([1,2,3], [1,4,5,6])
    array([[1, 4, 5, 6],
           [2, 1, 4, 5],
           [3, 2, 1, 4]])
    
    Regular usage:
    >>> I2 = np.eye(2)
    >>> block_toeplitz([1*I2,2*I2,3*I2], [1*I2,4*I2,5*I2,6*I2])
    array([[1, 0, 4, 0, 5, 0, 6, 0],
           [0, 1, 0, 4, 0, 5, 0, 6],
           [2, 0, 1, 0, 4, 0, 5, 0],
           [0, 2, 0, 1, 0, 4, 0, 5],
           [3, 0, 2, 0, 1, 0, 4, 0],
           [0, 3, 0, 2, 0, 1, 0, 4]])
    
    Usage with broadcasting of scalar blocks:
    >>> block_toeplitz([1*I2,2*I2,3*I2], [1,4,5,6])
    array([[1, 0, 4, 4, 5, 5, 6, 6],
           [0, 1, 4, 4, 5, 5, 6, 6],
           [2, 0, 1, 0, 4, 4, 5, 5],
           [0, 2, 0, 1, 4, 4, 5, 5],
           [3, 0, 2, 0, 1, 0, 4, 4],
           [0, 3, 0, 2, 0, 1, 4, 4]])
    '''
    c = [np.atleast_2d(ci) for ci in c]
    if r is None:
        r = [np.conj(ci) for ci in c]
    else:
        r = [np.atleast_2d(rj) for rj in r]
    
    mb,nb = c[0].shape
    dtype = (c[0]+r[0]).dtype
    m = len(c)
    n = len(r)
    
    A = np.zeros((m*mb, n*nb), dtype=dtype)
    
    for i in range(m):
        for j in range(n):
            # 1. select the Aij block from c or r:
            d = i-j
            if d>=0:
                Aij = c[d]
            else:
                Aij = r[-d]
            # 2. paste the block
            A[i*mb:(i+1)*mb, j*mb:(j+1)*mb] = Aij
    
    return A
