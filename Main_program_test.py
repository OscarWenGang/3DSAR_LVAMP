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


#%% 
xhat_ = training_stages[-1][1]
from scipy.io import loadmat
from scipy.io import savemat
import numpy as np
import time
D=loadmat('Y_exp.mat')
Ntrn = np.size(D['Y'],1)
Y = np.vstack((D['Y'].real,D['Y'].imag))
X = np.zeros((70,Ntrn))
Nbatch = 1000
flagend = False
index_start = 0
starttime = time.clock()
while not flagend:
    index_end = index_start + Nbatch
    if index_end > Ntrn:
        index_end = Ntrn
        flagend = True
    y = Y[:,index_start:index_end]
    xhat = sess.run(xhat_,feed_dict={prob.y_:y})
    X[:,index_start:index_end] = xhat
    index_start += Nbatch
    
endtime = time.clock()
t_lvamp = endtime-starttime
X = X[:35,:] + 1j*X[35:,:]
savemat('X_LVAMP_exp2.mat',{'X_lvamp' : X, 't_lvamp' : t_lvamp}, oned_as='column')
