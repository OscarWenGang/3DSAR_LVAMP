# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 15:24:55 2018

Main program

@author: admin
"""


from __future__ import division
from __future__ import print_function
#import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # BE QUIET!!!!
#import tensorflow as tf

#np.random.seed(1) # numpy is good about making repeatable output
#tf.set_random_seed(1) # on the other hand, this is basically useless (see issue 9171)

# import our problems, networks and training modules
import problem,network,train

# Create the basic problem structure.
snr='20'
prob = problem.CSSE(snr)

# build an LVAMP network to solve the problem and get the intermediate results so we can greedily extend and then refine(fine-tune)
layers = network.build_LVAMP(prob,T=8,shrink='pwlin')

# plan the learning
training_stages = train.setup_training(layers,prob,trinit=1e-4,refinements=(.5,.05),final_refine=.01)

# do the learning (takes a while)
sess = train.do_training(training_stages,prob,'LVAMP_un_' + snr + '.npz',ivl=20)
#sess = train.do_training(training_stages,prob,'LVAMP_nonun_' + snr + '.npz',ivl=20)

#%% time test & save results
#from scipy.io import savemat
#import time
#starttime = time.clock()
#sess.run(layers[-1][1],feed_dict = {prob.y_:prob.yval,prob.x_:prob.xval})
#sess.run(layers[-1][1],feed_dict = {prob.y_:prob.yval,prob.x_:prob.xval})
#sess.run(layers[-1][1],feed_dict = {prob.y_:prob.yval,prob.x_:prob.xval})
#sess.run(layers[-1][1],feed_dict = {prob.y_:prob.yval,prob.x_:prob.xval})
#xhat=sess.run(layers[-1][1],feed_dict = {prob.y_:prob.yval,prob.x_:prob.xval})
#endtime = time.clock()
#t_lvamp = (endtime-starttime)/5
#mse_lvamp = sess.run(training_stages[-1][3],feed_dict = {prob.y_:prob.yval,prob.x_:prob.xval})
#savemat('lvamp_un_snr' + snr + '.mat',{'X_lvamp' : xhat, 't_lvamp' : t_lvamp, 'mse_lvamp' : mse_lvamp}, oned_as='column')
##savemat('lvamp_nonun_snr' + snr + '.mat',{'X_lvamp' : xhat, 't_lvamp' : t_lvamp, 'mse_lvamp' : mse_lvamp}, oned_as='column')
