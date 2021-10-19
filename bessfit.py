# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 15:18:36 2021

@author: reonid

"""

import numpy as np
import scipy.special as spf

j0roots = spf.jn_zeros(0, 1000)


def _basis_func(n):
    nthroot = j0roots[n]

    def func(x):
        return spf.j0(x*nthroot)

    return func


basis_funcs = [_basis_func(i) for i in range(0, 20)]


def _fitf(x, *args):
    result = np.zeros_like(x)
    for i, a in enumerate(args):
        result = result + a*basis_funcs[i](x)
    return result


def bessfit1(x, p0):
    return p0*basis_funcs[0](x)


def bessfit2(x, p0, p1):
    return p0*basis_funcs[0](x) + p1*basis_funcs[1](x)


def bessfit3(x, p0, p1, p2): return _fitf(x, p0, p1, p2)
def bessfit4(x, p0, p1, p2, p3): return _fitf(x, p0, p1, p2, p3)
def bessfit5(x, p0, p1, p2, p3, p4): return _fitf(x, p0, p1, p2, p3, p4)
def bessfit6(x, p0, p1, p2, p3, p4, p5): return _fitf(x, p0, p1, p2, p3, p4, p5)
def bessfit7(x, p0, p1, p2, p3, p4, p5, p6): return _fitf(x, p0, p1, p2, p3, p4, p5, p6)
