# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 15:13:59 2018

Define the 3D-SAR problem

@author: admin
"""

from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
#from scipy.io import savemat
from scipy.io import loadmat
#from numba import autojit

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
        A_complex = np.exp(2j*self.k_mean/self.R_ref*np.outer(self.b_TR,self.s)).astype(np.complex64)
        A_complex = A_complex/np.sqrt(self.N_btr)
        A_up = np.hstack((A_complex.real,-A_complex.imag))
        A_down = np.hstack((A_complex.imag,A_complex.real))
        self.A = np.vstack((A_up,A_down))
        M,N = self.A.shape
        self.x_ = tf.placeholder( tf.float32,(N,None),name='x' )
        self.y_ = tf.placeholder( tf.float32,(M,None),name='y' )
        
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
            coef_t = np.random.randn(N_t) + 1j*np.random.randn(N_t)
            for c_t in range(N_t):
                sig = sig + coef_t[c_t] * np.exp(2j*self.k_mean/self.R_ref*self.b_TR*coord_x[c_t]).astype(np.complex64)
            sig = sig/np.sqrt(self.N_btr)
            
            gt = np.zeros(self.N_s) + 1j * np.zeros(self.N_s)
            index = np.round((coord_x + self.s_range/2)/(self.s[1]-self.s[0])).astype(np.int)
            for c_c in range(len(index)):
                gt[index[c_c]] += coef_t[c_c]
            sig_pow += np.sum(np.abs(sig)**2)
            Y[:,c_trn] = sig
            X[:,c_trn] = gt
        noise_pow = sig_pow/self.N_btr/N_batch/(10**(SNR/10))
        for c_trn in range(N_batch):
            noise = np.sqrt(noise_pow) * (np.random.randn(self.N_btr)/np.sqrt(2) + 1j*np.random.randn(self.N_btr)/np.sqrt(2))
            Y[:,c_trn] = Y[:,c_trn] + noise 
        Y = np.vstack((Y.real,Y.imag))
        X = np.vstack((X.real,X.imag))
        Y = Y.astype(np.float32)
        X = X.astype(np.float32)
        return X,Y
    
    def GetXY(self,batchS=500):
        endindex = self.start + batchS
        if endindex >self.Ntrn:
            self.start = 0
            endindex = batchS
            self.trn_order = np.random.permutation(self.Ntrn)
        X = self.xtrn[:,self.trn_order[self.start:endindex]]
        Y = self.ytrn[:,self.trn_order[self.start:endindex]]
        self.start += batchS
        return X, Y
        
def CSSE(snr='10'):
    prob = CSSE_GEN()
    
    D=loadmat('Valid_data_' + snr + '.mat')
#    D=loadmat('D:/nonuniform_b_TR/Valid_data_' + snr + '.mat')
    prob.xval = D['X']
    prob.yval = D['Y']   
    
#    prob.xval, prob.yval = prob.GenXY(N_tt=None,N_batch=1000)   # N_tt=3
#    prob.xinit, prob.yinit = prob.GenXY(N_tt=None,N_batch=1000)   # N_tt=3
    prob.Ntrn = 2000000  # 2000000
    
    D=loadmat('Train_data_' + snr + '.mat')
#    D=loadmat('D:/nonuniform_b_TR/Train_data_' + snr + '.mat')
    prob.xtrn = D['X']
    prob.ytrn = D['Y']   
    
#    prob.xtrn, prob.ytrn = prob.GenXY(N_tt=None,N_batch=prob.Ntrn)   # N_tt=3
    prob.start = 0
    prob.trn_order = np.random.permutation(prob.Ntrn)

#    D = {'X' : prob.xval, 'Y' : prob.yval, 'A' : prob.A}
#    savemat('Valid_data.mat',D,oned_as='column')
#    
#    D = {'X' : prob.xtrn, 'Y' : prob.ytrn}
#    savemat('Train_data.mat',D,oned_as='column')
    
    return prob
    