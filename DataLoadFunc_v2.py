#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data file readout in the appropriate form

@author: Ksenija Kovalenka
"""
import numpy as np

def load_data(file_path, npartitions, nkx, nky, nkz):
    nkpt = nkx*nky*nkz
    
    
    reals = np.zeros(shape=(nkpt,npartitions))
    complexs = np.zeros(shape=(nkpt,npartitions))
    alpha = np.zeros(shape=(nkpt,npartitions))
    
    
    data = open(file_path, 'r')
    
    for a in range(0, npartitions):
        for kz in range(0, nkpt):
            reals[kz,a] = float(data.read(10))
            complexs[kz,a] = float(data.read(13))
            alpha[kz,a] = float(data.read(7))
    data.close()
    
    data_tensor = np.zeros((npartitions, 2*nkz, nkx, nky))
    alpha_tensor = np.zeros((npartitions))
    phases_classification = np.zeros((npartitions,1))

    for i in range(npartitions):
        reals_1 = np.reshape(reals[::2,i], (11,11))
        reals_0 = np.reshape(reals[1::2,i], (11,11))
        complexs_1 = np.reshape(complexs[::2,i], (11,11))
        complexs_0 = np.reshape(complexs[1::2,i], (11,11))
        
        data_tensor[i, 0] = reals_1
        data_tensor[i, 1] = reals_0
        data_tensor[i, 2] = complexs_1
        data_tensor[i, 3] =  complexs_0
        alpha_tensor[i] = alpha[0,i]
        
        if alpha[0,i] < 0.77:
            phases_classification[i] = 0
        else:
            phases_classification[i] = 1
    #print(phases_classification[5000-5:5000+5])
    return data_tensor, phases_classification, alpha_tensor
#data, classifications = load_data('NN_data_equal_v0.dat', 10000, 11, 11, 2)


