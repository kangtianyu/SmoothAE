'''
Created on Apr 20, 2017

@author: tianyu
'''

import numpy as np
import theano.tensor as T

class Layer(object):
    def __init__(self, nodeNum, activation = None, grad = None):
        """
        Typical layer: 

        :type nodeNum: int
        :param nodeNum: number of nodes

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self._nodeNum = nodeNum
        self._activation = activation
        if grad is None:
            self._grad = lambda x:np.ones(len(x))
        else:
            self._grad = grad
        self._incomeLayer = []
        self._outgoLayer = []
        self._batch = 0
        self._avg_out = np.zeros(self._nodeNum)
        self._avg_err = np.zeros(self._nodeNum)
        # Average of previous epoch
        self._p_avg_out = np.zeros(self._nodeNum)
        self._p_avg_err = np.zeros(self._nodeNum)
        
        
    def addIncomeLayer(self,layer):
        self._incomeLayer.append(layer)
        
    def getIncomeLayers(self):
        return self._incomeLayer;
        
    def addOutgoLayer(self,layer):
        self._outgoLayer.append(layer)
        
    def getOutgoLayers(self):
        return self._outgoLayer;
        
    def computeOut(self,inputValues):
        if not inputValues.shape[0] == self._nodeNum:
            raise ValueError("Input number error: get " + str(inputValues.shape[0]) + ", expect " + str(self._nodeNum))
        if self._activation is None:
            self._out = inputValues
        else:
            self._out = self._activation(inputValues)
        self._avg_out += self._out
        self._batch += 1
        return self._out
    
    def setOutValues(self,values):
        self._out = values
    
    def getOutValues(self):
        if self._out is None:
            self._out = np.zeros(self._nodeNum)
        return self._out
    
    def getAverageOut(self):
        return self._p_avg_out
    
    def setErrors(self,err):
#         print(err)
#         err = values.eval()
        if not len(err) == self._nodeNum:
            raise ValueError("Input number error: get " + err.shape[1] + ", expect " + self._nodeNum)
        self._err = err
        self._avg_err += err
        
    def getErrors(self):
        return self._err
    
    def getAverageError(self):
        return self._p_avg_err
    
    def getGrad(self):
        return self._grad
    
    def setGrad(self,grad):
        self._grad = grad
        
    def update(self):
        self._p_avg_out = self._avg_out/self._batch
        self._p_avg_err = self._avg_err/self._batch
        self._avg_out = 0
        self._avg_err = 0
        self._batch = 0