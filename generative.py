# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 16:12:39 2017

@author: tuncb
"""

import numpy as np

def logsum(array):
    amax = np.max(array, axis=1);
    s = np.log(np.sum(np.exp(array - amax.reshape([len(amax),1])), axis=1)) + amax;

    return s
   
##############################################################################
#          Class that defines the generative model of a network              #
##############################################################################
class bayesianNetwork:
    def __init__(self, _A, wR=None, wM=0, Comps=None, multip=True, baseType=0, Mem=None, binMask=False):
        from scipy.special import gammaln

        self.A = _A*1.
        self.N = _A.shape[-1]
        self.mesoType = 'random'
        self.multiSclice = False
        self.sliceN = self.N
        self.Z = np.ones([self.N,1], dtype='float')
        self.c = np.zeros(self.N, dtype='float')
        self.K = 1
        self.wM = wM
        self.Module = None
        self.multip = multip
        self.baseType = baseType
        self.Comps = Comps
        self.wR = wR

        self.binMask = binMask

        if Mem is None:
            Mem = np.ones([self.N, 1])
        self.Mem = np.dot(Mem, Mem.T).astype('bool')

        if len(_A.shape) == 3:
            A = _A.mean(axis=0)
        else:
            A = _A*1.

        a = A.sum(axis=1)
        self.afl = gammaln(a+1)
        self.Afl = gammaln(A+1).sum(axis=1)

        self.nd = A.sum(axis=1).astype('float').reshape([self.N,1])

        self.initializeBaseModel(wR)

    def initializeBaseModel(self, wR):
        Comps = self.Comps
        #random model
        if Comps is not None:
            if self.baseType==0:
                B = np.ones([self.N,self.N])
                for i in range(len(Comps)):
                    if wR[i]==0:
                        continue
                    else:
                        comp = np.ma.array(Comps[i] , mask=(Comps[i] == 0))
                        B *= np.ma.filled(comp**wR[i], 1)
                        #B *= Comps[i]**wR[i] <--- wrong because it will multiply with 0
            elif self.baseType==1:
                B = np.zeros([self.N,self.N])
                for i in range(len(Comps)):
                    if wR[i]==0:
                        continue
                    else:
                        B += Comps[i]**wR[i]
                if B.max() < 1e-8:
                    B = np.ones([self.N,self.N])
            elif self.baseType==2:
                B = np.zeros([self.N,self.N])
                for i in range(len(Comps)):
                    if wR[i]==0:
                        continue
                    else:
                        B += Comps[i]*wR[i]
                if B.max() < 1e-8:
                    B = np.ones([self.N,self.N])
            elif self.baseType==3:
                B = np.zeros([self.N,self.N])
                for i in range(len(Comps)):
                    if wR[i]==0:
                        continue
                    else:
                        B += np.exp(Comps[i]*wR[i])
                if B.max() < 1e-8:
                    B = np.ones([self.N,self.N])
        else:
            B = np.ones([self.N,self.N])

        np.fill_diagonal(B, 0.)
        B[B<0] = 0.

        if self.binMask:
            mask = (self.A==0)
            B[mask] = 0.

        self.B = B.copy()
        self.Bnorm = self.nd * (B / (B.sum(axis=1).reshape([self.N, 1])+1e-16))

    def priorExpectations(self):
        if self.mesoType == 'random':
            B = self.Bnorm.copy()
        elif self.mesoType == 'empirical':
            B = self.A.copy()
        else:
            Z = self.Z
            wM = self.wM
            B = self.B.copy()

            #community model
            if self.mesoType == 'modular':
                Module = np.dot(Z, Z.T)
                B = B * (1. + wM*(2.*Module-1.))
            #core-periphery model
            elif self.mesoType == 'core':
                c = self.c.reshape([self.N,1])
                core = np.tile(c, (1, self.N)).astype('bool')
                core = core | core.T
                B = B * (1. + wM*(2.*core-1.))
            #modular core-periphery model
            elif self.mesoType == 'modular&core':
                Module = np.dot(Z, Z.T).astype('bool')
                c = self.c.reshape([self.N,1])
                core = np.tile(c, (1, self.N)).astype('bool')
                core = core | core.T
                modANDcore = (Module & core)
                B = B * (1. + wM*(2.*modANDcore-1.))
            #communities with connector hubs
            elif self.mesoType == 'modular&hub':
                Module = np.dot(Z, Z.T).astype('bool')
                c = self.c.reshape([self.N,1])
                Hub = np.tile(c, (1, self.N)).astype('bool')
                Group = self.Mem.copy()
                modORhub = (Module | (Hub & Group))
                B = B * (1. + wM*(2.*modORhub-1.))

            #normalize
            B[B<0] = 0.
            np.fill_diagonal(B, 0.)

            if self.binMask:
                mask = (self.A==0)
                B[mask] = 0.
            B = self.nd * (B / (B.sum(axis=1).reshape([self.N, 1])+1e-16))

        return B

    def likelihood(self, subject=None):
        from scipy.special import gammaln

        B = self.priorExpectations()

        if (len(self.A.shape) == 3) and (subject is not None):
            A = self.A[subject,:,:].copy()
        elif (len(self.A.shape) == 3):
            A = self.A.mean(axis=0)
        else:
            A = self.A.copy()

        b = B.sum(axis=1)
        tmp = gammaln(b)
        tmp[b<1e-16] = 0.
        bgl = tmp

        AB = A+B
        ab = AB.sum(axis=1)
        tmp = gammaln(ab)
        tmp[np.isnan(tmp)] = 0.
        tmp[ab<1e-16] = 0.
        abgl = tmp

        tmp = gammaln(AB) - gammaln(B)
        tmp[np.isnan(tmp)] = 0.
        tmp[tmp<1e-16] = 0.
        ABgl = (tmp).sum(axis=1)
        
        logL = self.afl - self.Afl + bgl - abgl + ABgl
        jlogL = np.sum(logL)

        if np.isnan(jlogL) :
            print "likelihood is NaN"

        return logL, jlogL

    def getRandomFit(self):
        mesotype = self.mesoType
        K = 1*self.K
        self.mesoType = 'random'
        self.K = 1

        if (len(self.A.shape) == 3):
            numSub=self.A.shape[0]
            logL = 0.
            for subject in range(numSub):
                _, _logL = self.likelihood(subject=subject)
                logL += _logL
            logL /= numSub
        else:
            _, logL = self.likelihood()

        self.mesoType = mesotype
        self.K = K

        return logL

    def getOptimalFit(self):
        mesotype = self.mesoType
        K = 1*self.K
        self.mesoType = 'empirical'
        self.K = 1
        _, logL = self.likelihood()
        self.mesoType = mesotype
        self.K = K

        return logL
    
