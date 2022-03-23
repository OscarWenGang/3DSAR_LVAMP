# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 08:36:39 2018

Generate Training and testing data under different SNRs

@author: admin
"""

from __future__ import division
from __future__ import print_function
import numpy as np
#import tensorflow as tf
from scipy.io import savemat
from scipy.io import loadmat


class CSSE_GEN(object):
    e0 = 8.85e-12
    u0 = 4*np.pi*1e-7
    c = 1/np.sqrt(e0*u0)     # speed of light
    def __init__(self):
        f0 = 35e9   # carrier frequency
        lamda = self.c/f0
        self.k_mean = f0*2*np.pi/self.c
        ############ uniform ############
        dbtr = 0.002
        self.b_TR = np.arange(-0.03,0.03+dbtr,dbtr)

        self.N_btr = len(self.b_TR)
        self.R_ref = 3.3
        self.s_range = 0.8
        self.N_t_max = 4
        dks = 4*np.pi*dbtr/lamda/self.R_ref
        s_alise = 2*np.pi/dks
        SR_ratio = 10
        self.s = np.linspace(-self.s_range/2,self.s_range/2,np.int(np.round(SR_ratio*self.N_btr*self.s_range/s_alise)))
        self.N_s = len(self.s)
#        A_complex = np.exp(2j*self.k_mean/self.R_ref*np.outer(self.b_TR,self.s)).astype(np.complex64)
#        A_complex = A_complex/np.sqrt(self.N_btr)
#        A_up = np.hstack((A_complex.real,-A_complex.imag))
#        A_down = np.hstack((A_complex.imag,A_complex.real))
#        self.A = np.vstack((A_up,A_down))
        
    def GenXY(self,N_tt=None,N_batch=100000,SNR=10):
        Y = np.zeros((self.N_btr,N_batch)) + 1j * np.zeros((self.N_btr,N_batch))
        X = np.zeros((self.N_s,N_batch)) + 1j * np.zeros((self.N_s,N_batch))
        N_t = np.zeros(1)
        if N_tt is not None:
            N_t = N_tt
        sig_pow = 0
        for c_trn in range(N_batch):
            sig = np.zeros(self.N_btr) + 1j * np.zeros(self.N_btr)
            if N_tt is None:
                N_t = np.int(np.ceil(self.N_t_max * np.random.uniform()))
            coord_x = self.s_range * np.random.uniform(size=N_t) - self.s_range/2
            coord_x = coord_x *0.8   # 6.26：为了提升对实验数据的视觉效果而加
            coef_t = np.random.randn(N_t) + 1j*np.random.randn(N_t)
            for c_t in range(N_t):
                sig = sig + coef_t[c_t] * np.exp(2j*self.k_mean/self.R_ref*self.b_TR*coord_x[c_t]).astype(np.complex64)
            
            # noise = np.sqrt(10**((3-self.SNR)/10)) * (np.random.randn(self.N_btr)/np.sqrt(2) + 1j*np.random.randn(self.N_btr)/np.sqrt(2))
            # sig = sig + noise
            sig = sig/np.sqrt(self.N_btr)
            
            gt = np.zeros(self.N_s) + 1j * np.zeros(self.N_s)
            index = np.round((coord_x + self.s_range/2)/(self.s[1]-self.s[0])).astype(np.int)
            for c_c in range(len(index)):
                gt[index[c_c]] += coef_t[c_c]
            sig_pow += np.sum(np.abs(sig)**2)
            Y[:,c_trn] = sig
            X[:,c_trn] = gt
        noise_pow = sig_pow/self.N_btr/N_batch/(10**(SNR/10))
#        noise = np.sqrt(noise_pow) * (np.random.randn(self.N_btr,N_batch)/np.sqrt(2) + 1j*np.random.randn(self.N_btr,N_batch)/np.sqrt(2))
#        Y = Y + noise
        for c_trn in range(N_batch):
            noise = np.sqrt(noise_pow) * (np.random.randn(self.N_btr)/np.sqrt(2) + 1j*np.random.randn(self.N_btr)/np.sqrt(2))
            Y[:,c_trn] = Y[:,c_trn] + noise 
        Y = np.vstack((Y.real,Y.imag))
        X = np.vstack((X.real,X.imag))
        Y = Y.astype(np.float32)
        X = X.astype(np.float32)
        return X,Y
    
prob = CSSE_GEN()

#%% test data generation
#xval, yval = prob.GenXY(N_tt=None,N_batch=1000,SNR=-5)
#D = {'X' : xval, 'Y' : yval}
#savemat('Valid_data_-5.mat',D,oned_as='column')

#np.random.seed(1)
#xval, yval = prob.GenXY(N_tt=None,N_batch=1000,SNR=0)
#D = {'X' : xval, 'Y' : yval}
#savemat('Valid_data_0.mat',D,oned_as='column')
#
#np.random.seed(1)
#xval, yval = prob.GenXY(N_tt=None,N_batch=1000,SNR=1)
#D = {'X' : xval, 'Y' : yval}
#savemat('Valid_data_1.mat',D,oned_as='column')
#
#np.random.seed(1)
#xval, yval = prob.GenXY(N_tt=None,N_batch=1000,SNR=2)
#D = {'X' : xval, 'Y' : yval}
#savemat('Valid_data_2.mat',D,oned_as='column')
#
#np.random.seed(1)
#xval, yval = prob.GenXY(N_tt=None,N_batch=1000,SNR=3)
#D = {'X' : xval, 'Y' : yval}
#savemat('Valid_data_3.mat',D,oned_as='column')
#
#np.random.seed(1)
#xval, yval = prob.GenXY(N_tt=None,N_batch=1000,SNR=4)
#D = {'X' : xval, 'Y' : yval}
#savemat('Valid_data_4.mat',D,oned_as='column')
#
#np.random.seed(1)
#xval, yval = prob.GenXY(N_tt=None,N_batch=1000,SNR=5)
#D = {'X' : xval, 'Y' : yval}
#savemat('Valid_data_5.mat',D,oned_as='column')
#
#np.random.seed(1)
#xval, yval = prob.GenXY(N_tt=None,N_batch=1000,SNR=6)
#D = {'X' : xval, 'Y' : yval}
#savemat('Valid_data_6.mat',D,oned_as='column')
#
#np.random.seed(1)
#xval, yval = prob.GenXY(N_tt=None,N_batch=1000,SNR=7)
#D = {'X' : xval, 'Y' : yval}
#savemat('Valid_data_7.mat',D,oned_as='column')
#
#np.random.seed(1)
#xval, yval = prob.GenXY(N_tt=None,N_batch=1000,SNR=8)
#D = {'X' : xval, 'Y' : yval}
#savemat('Valid_data_8.mat',D,oned_as='column')
#
#np.random.seed(1)
#xval, yval = prob.GenXY(N_tt=None,N_batch=1000,SNR=9)
#D = {'X' : xval, 'Y' : yval}
#savemat('Valid_data_9.mat',D,oned_as='column')

#np.random.seed(1)
#xval, yval = prob.GenXY(N_tt=1,N_batch=1000,SNR=10)
#D = {'X' : xval, 'Y' : yval}
#savemat('Valid_data_10.mat',D,oned_as='column')

#np.random.seed(1)
#xval, yval = prob.GenXY(N_tt=None,N_batch=1000,SNR=11)
#D = {'X' : xval, 'Y' : yval}
#savemat('Valid_data_11.mat',D,oned_as='column')
#
#np.random.seed(1)
#xval, yval = prob.GenXY(N_tt=None,N_batch=1000,SNR=12)
#D = {'X' : xval, 'Y' : yval}
#savemat('Valid_data_12.mat',D,oned_as='column')
#
#np.random.seed(1)
#xval, yval = prob.GenXY(N_tt=None,N_batch=1000,SNR=13)
#D = {'X' : xval, 'Y' : yval}
#savemat('Valid_data_13.mat',D,oned_as='column')
#
#np.random.seed(1)
#xval, yval = prob.GenXY(N_tt=None,N_batch=1000,SNR=14)
#D = {'X' : xval, 'Y' : yval}
#savemat('Valid_data_14.mat',D,oned_as='column')
#
np.random.seed(1)
xval, yval = prob.GenXY(N_tt=1,N_batch=1000,SNR=20)
D = {'X' : xval, 'Y' : yval}
savemat('Valid_data_20.mat',D,oned_as='column')

#%% training data

#xval, yval = prob.GenXY(N_tt=None,N_batch=2000000,SNR=-5)
#D = {'X' : xval, 'Y' : yval}
#savemat('Train_data_-5.mat',D,oned_as='column')
#
#xval, yval = prob.GenXY(N_tt=None,N_batch=2000000,SNR=0)
#D = {'X' : xval, 'Y' : yval}
#savemat('Train_data_0.mat',D,oned_as='column')
#
#xval, yval = prob.GenXY(N_tt=None,N_batch=2000000,SNR=5)
#D = {'X' : xval, 'Y' : yval}
#savemat('Train_data_5.mat',D,oned_as='column')
#
#xval, yval = prob.GenXY(N_tt=1,N_batch=2000000,SNR=10)
#D = {'X' : xval, 'Y' : yval}
#savemat('Train_data_10.mat',D,oned_as='column')
#
xval, yval = prob.GenXY(N_tt=1,N_batch=2000000,SNR=20)
D = {'X' : xval, 'Y' : yval}
savemat('Train_data_20.mat',D,oned_as='column')



