'''
Created on Apr 28, 2017

@author: tianyu
'''
import time
import Layer
import theano
import numpy as np
import numpy.random as rng
import theano.tensor as T
import theano.tensor.nnet as nn
import theano.sparse.basic as S
import theano.sparse
import scipy.sparse as sp
import Datasets

class SmoothLayer(object):
    def __init__(self, inputSize):
        """Initialize the parameters for the Smooth Auto Encoder

        :type inputData: theano.tensor.TensorType
        :param inputData: symbolic variable that describes the input of the
        architecture (one minibatch)

        """
        
        # epoch num
        self._epochNum = 0
        
        """ Initalize """
        self._inputNum = inputSize
        # Layers
        self._smoothLayer = Layer.Layer(nodeNum = self._inputNum)
        self._contextLayer = Layer.Layer(nodeNum = self._inputNum)
        
        self._layers = [self._smoothLayer,self._contextLayer]
        
        # Connections
        # input layer to smooth Layer
        # 1-self._alpha
#         self._alpha = theano.shared(0.5)
        self._alpha = 0.6
        
        # smooth layer to context layer
        # 1
        
        # context layer to smooth layer
        # ADJ_MAT * self._alpha
        
    # one epoch
    def learn(self,inputData):
        if not len(inputData[0]) == self._inputNum:
            raise ValueError("Input number error: get " + str(len(inputData[0])) + ", expect " + str(self._inputNum))
        self._miniBatchSize = len(inputData)
        
        count = 0
        
        for sample in inputData:
            count += 1
            
            """ define neural network """
            """ Forward """
            # define input layer        
            x = sample
            # define smooth layer
            # smooth function is f_(t+1) = alpha * ADJ_MAT * f_t + (1-alpha) * f_0
            
            smIn =  self._alpha * S.dot(T.as_tensor_variable(self._contextLayer.getAverageOut()),Datasets.W) + np.multiply(x,1-self._alpha)
            smOut = self._smoothLayer.computeOut(smIn.eval())
#             Datasets.dmpnp("s_out_" + str(self._epochNum) + "_" + str(count), self._smoothLayer.getOutValues())

            # update context layer
            self._contextLayer.computeOut(self._smoothLayer.getOutValues())
            
        
        """ update network"""
        
        for layer in self._layers:
            layer.update()
        
    def train(self,inputData):
        while True:
            t0 = time.time()
            self._epochNum += 1
            Datasets.dmpnp("c_out_" + str(self._epochNum), self._contextLayer.getAverageOut())
            self.learn(inputData)
            Datasets.dmpnp("s_out_" + str(self._epochNum), self._smoothLayer.getAverageOut())
            Datasets.log("epoch " + str(self._epochNum) + "(" + str(time.time()-t0) + "s)")
#             self.save("_" + str(epoch_num))
            # Stopping rules
            if(self._epochNum > 100):
                break
            
    def save(self,str=""):
        Datasets.dmpnp("s_out" + str, self._smoothLayer.getAverageOut())
        Datasets.dmpnp("c_out" + str, self._contextLayer.getAverageOut())
        

def test_SmoothLayer():    
    data = Datasets.samples
    network = SmoothLayer(len(Datasets.labels))
    network.train(data)

if __name__ == '__main__':
    test_SmoothLayer()