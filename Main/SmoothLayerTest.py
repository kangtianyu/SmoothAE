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

def quickMul(a,b):
    li = np.transpose(np.nonzero(a))
    rows = []
    columns = []
    data = []
    size =  theano.sparse.basic.csm_shape(a)[0].eval()
    for i in li:
        row = i[0]
        column = i[1]
        rows.append(row)
        columns.append(column)
        data.append(a[row,column]*b[row,column])
    return sp.csr_matrix((data,(rows,columns)), shape=(size, size))

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
#         self._contextLayer.setAverageOut(S.structured_dot(T.as_tensor_variable([rng.rand(self._inputNum)]),Datasets.W).eval()[0])
        
        self._layers = [self._smoothLayer,self._contextLayer]
        
        # Connections
        # input layer to smooth Layer
        # 1-self._alpha
#         self._alpha = theano.shared(0.5)
        self._alpha = 0.5
        self._alpha_change = 0
        self._talpha = 0.9
        self._eta = 3e1
        self._err = 0;
        
        # smooth layer to context layer
        # 1
        
        # context layer to smooth layer
        # ADJ_MAT * self._alpha
        
    # one epoch
    def learn(self,inputData):
        if not len(inputData[0]) == self._inputNum:
            raise ValueError("Input number error: get " + str(len(inputData[0])) + ", expect " + str(self._inputNum))
        self._miniBatchSize = len(inputData)
        
        print(self._alpha,self._alpha_change,self._err)
        
        count = 0
        self._err = 0;
        self._alpha_change = 0
        
        for sample in inputData:
            count += 1
            
            """ define neural network """
            """ Forward """
            # define input layer        
            x = sample
            # define smooth layer
            # smooth function is f_(t+1) = alpha * ADJ_MAT * f_t + (1-alpha) * f_0
            
            smIn =  self._alpha * T.transpose(S.structured_dot(Datasets.W,T.transpose(T.as_tensor_variable([self._contextLayer.getAverageOut()])))).eval()[0] + np.multiply(x,1-self._alpha)
            smOut = self._smoothLayer.computeOut(smIn)
#             Datasets.dmpnp("s_out_" + str(self._epochNum) + "_" + str(count), self._smoothLayer.getOutValues())

            # update context layer
            self._contextLayer.computeOut(self._smoothLayer.getOutValues())
            
            # err
            terr = ((self._y - smOut) ** 2).sum()
            self._err += terr
            
            """ Back propagation """
            # error for out layer
            self._smoothLayer.setErrors(
                (self._y - self._smoothLayer.getOutValues()) 
                * self._smoothLayer.getGrad()(self._smoothLayer.getOutValues()))
            # error for context layer
            self._contextLayer.setErrors(
                self._alpha
                * S.structured_dot(T.as_tensor_variable([self._smoothLayer.getErrors()]),Datasets.W).eval()[0]
                * self._contextLayer.getGrad()(self._contextLayer.getOutValues()))
            
            
            alpha_change_c = S.sp_sum(quickMul(Datasets.W,S.structured_dot(S.transpose(sp.csr_matrix(np.asarray([self._contextLayer.getAverageOut()]))), sp.csr_matrix(np.asarray([self._smoothLayer.getErrors()]))).eval())).eval()/Datasets.W_SUM         
            alpha_change_i = np.sum(np.multiply(x,self._smoothLayer.getErrors()))/self._inputNum
            self._alpha_change += alpha_change_c - alpha_change_i
        
        """ update network"""
        
        for layer in self._layers:
            layer.update()
            
        self._alpha += self._eta * self._alpha_change
        
    def train(self,inputData):
        x0 = inputData[0]
        xt = inputData[0]
        while True:
            xt1 = self._talpha * T.transpose(S.structured_dot(Datasets.W,T.as_tensor_variable(np.transpose([xt])))).eval()[0] + np.multiply(x0,1-self._talpha)
            if np.sum((xt1-xt)**2)<1e-100:
                break
            xt = xt1
        self._y =np.copy(xt1)
        Datasets.dmpnp("y",self._y)
        
        while True:
            t0 = time.time()
            self._epochNum += 1
            self.learn(inputData)
            Datasets.log("epoch " + str(self._epochNum) + "(" + str(time.time()-t0) + "s)")
#             self.save("_" + str(epoch_num))
            # Stopping rules
            if(np.abs(self._alpha_change) <1e-17):
                break
            
        Datasets.dmpnp("x",self._smoothLayer.getAverageOut())
        
    def save(self,str=""):
        Datasets.dmpnp("s_out" + str, self._smoothLayer.getAverageOut())
        Datasets.dmpnp("c_out" + str, self._contextLayer.getAverageOut())
        

def test_SmoothLayer():    
    data = Datasets.samples[0:1]
    network = SmoothLayer(len(Datasets.labels))
    network.train(data)

if __name__ == '__main__':
    test_SmoothLayer()