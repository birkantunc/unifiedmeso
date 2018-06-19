# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 16:12:39 2017

@author: tuncb
"""

import numpy as np
from generative import bayesianNetwork
from optimization import simulated_annealing, annealing_model_single, annealing_model_hybrid

def initializeModules(N, K):
    if K==1:
        Z = np.ones([N,1], dtype='float')
    else:
        Z = np.zeros([N,K])
        for n in range(N):
            idx = np.random.randint(0, K)
            Z[n, idx] = 1

    return Z
    
##############################################################################
#                 functions to infere meso-scale parameters                  #
##############################################################################
def meso_structure_single(A, mesoType='modular', K=2, fullOptimization=True, xprior=None, bayN=None, binMask=True):   
    N = A.shape[0]
    
    wM = 0.5
    if bayN is None:
        bayN = bayesianNetwork(A, wR=None, wM=wM, Comps=None, binMask=binMask)        
    logLRand = bayN.getRandomFit()
    logLOpt = bayN.getOptimalFit()
    bayN.mesoType = mesoType
    if mesoType == 'modular':
        bayN.K = K
    
    if xprior is None:
        if mesoType == 'modular':
            x0 = initializeModules(N, K)
        elif mesoType == 'core':
            x0 = np.random.randint(0,2,N)
    else:
        x0 = xprior
   
    model = annealing_model_single(bayN)
    
    #check if the scale paramater (and others) will be optimized
    if fullOptimization:    
        opt = simulated_annealing(model, separate=True)

        #check if a base model of connectivity (geometric, genetic etc.) is defined
        if (bayN.Comps is None) or (bayN.wR is None):
            parameters = [wM]
        else:
            parameters = [wM] + bayN.wR

        state0 = [x0] + parameters
        state, f0 = opt.anneal(state0)
        x = state[0]
        wM = state[1]
    else:
        opt = simulated_annealing(model, separate=False)
        state0 = [x0]        
        state, f0 = opt.anneal(state0)
        x = state[0]

    return [logLRand,f0,logLOpt], x, wM
    
def meso_structure_hybrid(A, mesoType='modular&core', K=2, fullOptimization=True, xprior=None, bayN=None, binMask=True):   
    N = A.shape[0]
    
    wM = 0.5
    if bayN is None:
        bayN = bayesianNetwork(A, wR=None, wM=wM, Comps=None, binMask=binMask)        
    logLRand = bayN.getRandomFit()
    logLOpt = bayN.getOptimalFit()
    bayN.mesoType = mesoType
    bayN.K = K
    
    x0 = xprior
    if xprior[0] is None:
        x0[0] = initializeModules(N, K)
    if xprior[1] is None:
        x0[1] = np.random.randint(0,2,N)
   
    model = annealing_model_hybrid(bayN)
    
    #check if the scale paramater (and others) will be optimized
    if fullOptimization:    
        opt = simulated_annealing(model, separate=True)

        #check if a base model of connectivity (geometric, genetic etc.) is defined
        if (bayN.Comps is None) or (bayN.wR is None):
            parameters = [wM]
        else:
            parameters = [wM] + bayN.wR

        state0 = x0 + parameters
        state, f0 = opt.anneal(state0)
        Z = state[0]
        c = state[1]
        wM = state[2]
    else:
        opt = simulated_annealing(model, separate=False)
        state0 = [x0]        
        state, f0 = opt.anneal(state0)
        Z = state[0]
        c = state[1]

    return [logLRand,f0,logLOpt], Z, c, wM  

##############################################################################
#                           functions for end-users                          #
##############################################################################
def communities(A, K=2, fullOptimization=True, Zprior=None, bayN=None, binMask=True):
    logs, Z, wM = meso_structure_single(A, mesoType='modular', K=K, fullOptimization=fullOptimization, xprior=Zprior, bayN=bayN, binMask=binMask)    
    
    return logs, Z.argmax(axis=1), wM 

def core_periphery(A, fullOptimization=True, cprior=None, bayN=None, binMask=True):
    logs, c, wM = meso_structure_single(A, mesoType='core', fullOptimization=fullOptimization, xprior=cprior, bayN=bayN, binMask=binMask)    
    
    return logs, c, wM
    
def communities_of_core_periphery(A, K=2, fullOptimization=True, Zprior=None, cprior=None, bayN=None, binMask=True):
    logs, Z, c, wM = meso_structure_hybrid(A, mesoType='modular&core', K=K, fullOptimization=fullOptimization, xprior=[Zprior, cprior], bayN=bayN, binMask=binMask)    
    
    return logs, Z.argmax(axis=1), c, wM
    
