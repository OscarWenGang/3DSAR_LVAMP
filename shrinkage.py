#!/usr/bin/python

import tensorflow as tf

__doc__ = """
This file contains various separable shrinkage functions for use in TensorFlow.
All functions perform shrinkage toward zero on each elements of an input vector
    r = x + w, where x is sparse and w is iid Gaussian noise of a known variance rvar

All shrink_* functions are called with signature

    xhat,dxdr = func(r,rvar,theta)

Hyperparameters are supplied via theta (which has length ranging from 1 to 5)
    shrink_soft_threshold : 1 or 2 parameters
    shrink_bgest : 2 parameters
    shrink_expo : 3 parameters
    shrink_spline : 3 parameters
    shrink_piecwise_linear : 5 parameters

A note about dxdr:
    dxdr is the per-column average derivative of xhat with respect to r.
    So if r is in Real^(NxL),
    then xhat is in Real^(NxL)
    and dxdr is in Real^L
"""

def shrink_piecwise_linear(r,rvar,theta):
    """Implement the piecewise linear shrinkage function.
        With minor modifications and variance normalization.
        theta[...,0] : abscissa of first vertex, scaled by sqrt(rvar)
        theta[...,1] : abscissa of second vertex, scaled by sqrt(rvar)
        theta[...,2] : slope from origin to first vertex
        theta[''',3] : slope from first vertex to second vertex
        theta[...,4] : slope after second vertex
    """
    ab0 = theta[...,0]
    ab1 = theta[...,1]
    sl0 = theta[...,2]
    sl1 = theta[...,3]
    sl2 = theta[...,4]

    # scale each column by sqrt(rvar)
    scale_out = tf.sqrt(rvar)
    scale_in = 1/scale_out
    rs = tf.sign(r*scale_in)
    ra = tf.abs(r*scale_in)

    # split the piecewise linear function into regions
    rgn0 = tf.to_float( ra<ab0)
    rgn1 = tf.to_float( ra<ab1) - rgn0
    rgn2 = tf.to_float( ra>=ab1)
    xhat = scale_out * rs*(
            rgn0*sl0*ra +
            rgn1*(sl1*(ra - ab0) + sl0*ab0 ) +
            rgn2*(sl2*(ra - ab1) +  sl0*ab0 + sl1*(ab1-ab0) )
            )
    dxdr =  sl0*rgn0 + sl1*rgn1 + sl2*rgn2
    dxdr = tf.reduce_mean(dxdr,0)
    return (xhat,dxdr)

def get_shrinkage_function(name):
	"retrieve a shrinkage function and some (probably awful) default parameter values"
	try:
		return {
			'pwlin':(shrink_piecwise_linear, (2,4,0.1,1.5,.95) ),
		}[name]
	except KeyError.ke:
		raise ValueError('unrecognized shrink function %s' % name)

