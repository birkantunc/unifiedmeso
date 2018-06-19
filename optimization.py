# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 16:12:39 2017

@author: tuncb
"""

import numpy as np
    
##############################################################################
#                    functions to be used in optimization                    #
##############################################################################
class simulated_annealing():
    def __init__(self, model, threshold=1e-3, separate=False):
        self.model = model
        self.threshold = threshold
        self.separate = separate

    def accept(self, e0, e1, T):
        return np.exp((e0-e1)/T)#1.*(e1<e0)#

    def anneal(self, s0, numInIt=1000):
        e0 = self.model.energy(s0)
        T = 1.0
        Tmin = 0.00001
        alpha = 0.9
        counter = 0
        stopNow = False
        previous = 0
        while T > Tmin:
            if stopNow:
                break

            for i in range(numInIt):
                counter += 1
                if counter > numInIt:
                    stopNow = True
                    break

                if self.separate:
                    for par in range(len(s0)):
                        _s1 = self.model.update(s0)
                        s1 = s0[:]
                        s1[par] = _s1[par]
                        e1 = self.model.energy(s1)
                        ap = self.accept(e0, e1, T)
                        if (ap > np.random.rand()):# and (np.abs(e1-e0)>1e-8):
                            s0[par] = s1[par]
                            e0 = e1
                            counter = 0
                else:
                    s1 = self.model.update(s0)
                    e1 = self.model.energy(s1)
                    ap = self.accept(e0, e1, T)
                    if (ap > np.random.rand()):# and (np.abs(e1-e0)>1e-8):
                        s0 = s1
                        e0 = e1
                        counter = 0

            print("%0.5f: %0.6f" % (T, e0))

            #check convergence
            current = e0
            diff = np.abs(current-previous)
            previous = current
            if diff < self.threshold:
                if T < 0.2:
                    stopNow = True
                else:
                    s0 = self.model.update(s0)
                    e0 = self.model.energy(s0)
                    print("Making a small perturbation to prevent early termination")

            T = T*alpha

        return s0, -1*e0

class annealing_model_single():
    def __init__(self, bayN):
        self.bayN = bayN
        self.mesoType = bayN.mesoType

    def update(self, state):
        if self.mesoType == 'modular':
            Z = state[0].copy()
            K = Z.shape[1]
            indN = np.random.randint(0, Z.shape[0])
            indK = np.random.randint(0, K)
            Z[indN, :] = 0.
            Z[indN, indK] = 1.
            newState = [Z]
        elif self.mesoType == 'core':
            c = state[0].copy()
            ind = np.random.randint(0, len(c))
            c[ind] = 1. - c[ind]
            newState = [c]
        
        #parameters
        for i in range(1, len(state)):
            update = np.random.normal(0, 0.01)
            newState += [max(0., state[i] + update)]

        return newState

    def energy(self, state):
        if self.mesoType == 'core':
            self.bayN.c = state[0].copy()
        elif self.mesoType == 'modular':
            self.bayN.Z = state[0].copy()
        
        #check if the scale parameter will be optimized
        if len(state) > 1:
            self.bayN.wM = state[1]

        #check if a base model of connectivity (geometric, genetic etc.) is defined
        if len(state) > 2:
            self.bayN.initializeBaseModel(wR=state[2:])

        _, f = self.bayN.likelihood()

        return -1*f

class annealing_model_hybrid():
    def __init__(self, bayN):
        self.bayN = bayN

    def update(self, state):
        #subnetworks
        Z = state[0].copy()
        K = Z.shape[1]
        indN = np.random.randint(0, Z.shape[0])
        indK = np.random.randint(0, K)
        Z[indN, :] = 0.
        Z[indN, indK] = 1.
        newState = [Z]

        #cores
        c = state[1].copy()
        ind = np.random.randint(0, len(c))
        c[ind] = 1. - c[ind]
        newState += [c]

        #parameters
        for i in range(2, len(state)):
            update = np.random.normal(0, 0.01)
            newState += [max(0., state[i] + update)]

        return newState

    def energy(self, state):
        #subnetworks
        Z = state[0].copy()
        self.bayN.Z = Z
        #cores
        c = state[1].copy()
        self.bayN.c = c
        
        #check if the scale parameter will be optimized
        if len(state) > 2:
            self.bayN.wM = state[2]

        #check if a base model of connectivity (geometric, genetic etc.) is defined
        if len(state) > 3:
            self.bayN.initializeBaseModel(wR=state[3:])

        _, f = self.bayN.likelihood()

        return -1*f
    
