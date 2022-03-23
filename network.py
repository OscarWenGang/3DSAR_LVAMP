# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 15:09:38 2018

Build the LVAMP network

@author: admin
"""

from __future__ import division
from __future__ import print_function
import numpy as np
#import numpy.linalg as la

import tensorflow as tf
import shrinkage

def build_LVAMP(prob,T,shrink):
    """ Builds the non-SVD (i.e. dense) parameterization of LVAMP
    and returns a list of trainable points(name,xhat_,newvars)
    """
    eta,theta_init = shrinkage.get_shrinkage_function(shrink)
    layers=[]
    A = prob.A
    M,N = A.shape

    Hinit = A.T   # np.matmul(prob.xinit,la.pinv(prob.yinit) )
    H_ = tf.Variable(Hinit,dtype=tf.float32,name='H0')
    xhat_lin_ = tf.matmul(H_,prob.y_)
    layers.append( ('Linear',xhat_lin_,None) )

    vs_def = np.array(1,dtype=np.float32)
    theta_init = np.tile( theta_init ,(N,1,1))
    vs_def = np.tile( vs_def ,(N,1))

    theta_ = tf.Variable(theta_init,name='theta0',dtype=tf.float32)
    vs_ = tf.Variable(vs_def,name='vs0',dtype=tf.float32)
    rhat_nl_ = xhat_lin_
    rvar_nl_ = vs_ * tf.reduce_sum(prob.y_*prob.y_,0)/N

    xhat_nl_,alpha_nl_ = eta(rhat_nl_ , rvar_nl_,theta_ )
    layers.append( ('LVAMP-{0} T={1}'.format(shrink,1),xhat_nl_, None ) )
    eps = .5/N
    alpha_nl_ = tf.maximum(eps,tf.minimum(1-eps, alpha_nl_ ) )
#    GT_image_t = prob.x_[:,0:20]
#    GT_image = tf.reshape(GT_image_t,[1,156,20,1])
#    tf.summary.image('Groundtruth',GT_image,1)
    
    for t in range(1,T):
        # alpha_nl_ = tf.reduce_mean( alpha_nl_,axis=0) # each col average dxdr  # 6.13：dxdr在eta函数内部已经过reduce_mean

        gain_nl_ = 1.0 /(1.0 - alpha_nl_)
        rhat_lin_ = gain_nl_ * (xhat_nl_ - alpha_nl_ * rhat_nl_)
        rvar_lin_ = rvar_nl_ * alpha_nl_ * gain_nl_

        H_ = tf.Variable(Hinit,dtype=tf.float32,name='H'+str(t))
        G_ = tf.Variable(.9*np.identity(N),dtype=tf.float32,name='G'+str(t))
        xhat_lin_ = tf.matmul(H_,prob.y_) + tf.matmul(G_,rhat_lin_)

        layers.append( ('LVAMP-{0} lin T={1}'.format(shrink,1+t),xhat_lin_, (H_,G_) ) )

#        alpha_lin_ = tf.expand_dims(tf.diag_part(G_),1)  # 6.14：根据论文中对VAMP算法的描述对此值的定义进行修正
        alpha_lin_ = tf.reduce_mean(tf.diag_part(G_))

#        eps = .5/N
        alpha_lin_ = tf.maximum(eps,tf.minimum(1-eps, alpha_lin_ ) )

        vs_ = tf.Variable(vs_def,name='vs'+str(t),dtype=tf.float32)

        gain_lin_ = vs_ * 1.0/(1.0 - alpha_lin_)
        gain_lin_ = tf.maximum(1e-12,gain_lin_)  # 6.15：debug后分析发现，需对该值进行限制以使 rvar_nl_ 大于0
        
        rhat_nl_ = gain_lin_ * (xhat_lin_ - alpha_lin_ * rhat_lin_)
        rvar_nl_ = rvar_lin_ * alpha_lin_ * gain_lin_

        theta_ = tf.Variable(theta_init,name='theta'+str(t),dtype=tf.float32)

        xhat_nl_,alpha_nl_ = eta(rhat_nl_ , rvar_nl_,theta_ )
        alpha_nl_ = tf.maximum(eps,tf.minimum(1-eps, alpha_nl_ ) )
        layers.append( ('LVAMP-{0}  nl T={1}'.format(shrink,1+t),xhat_nl_, (vs_,theta_,) ) )
        
    Out_image_t = xhat_nl_[:,0:20]
    Out_image = tf.reshape(Out_image_t,[1,35,40,1])
    tf.summary.image( 'Output',Out_image,1 )
        
    return layers
