'''
Created on May 6, 2017

@author: tianyu
'''

import numpy as np
import numpy.random as rng
import theano
import theano.tensor as T
import theano.tensor.nnet as nn
import theano.tensor.nlinalg as nlg
import theano.sparse.basic as S
import scipy.sparse as sp
import time
import Layer
import os

class AE(object):
    
    def __init__(self, inputSize, L, learningRule = None, costFunction = None):
        """Initialize the parameters for the Smooth Auto Encoder

        :type inputData: theano.tensor.TensorType
        :param inputData: symbolic variable that describes the input of the
        architecture (one minibatch)

        """
        self._hc = 1.0
        self._lhc = 1.0
        self.L = L
        """ Hyper-parameters """
        # Sub-space dimension         
        self._k = 2
        
        # Learning rate
        self._eta = 2e1
        self._eta_increase_ratio = 1.05
        self._eta_max = 2e-1
        
        # Regularization penalty
        self._lamb = 1e-1
        
        # epoch num
        self._epochNum = 0
        
        """ Initalize """
        self._inputNum = inputSize
        
        # user define grad because auto generated grad not working correctly
        # grad of relu
        d_relu = lambda x: np.greater_equal(x,0)*1.
        
        sig = lambda x: 1/(1+np.exp(-x))
        
        def d_sig(x):
            tmp = sig(x)
            return tmp * (1 - tmp)
        
        # Layers
        self._wmatrixLayer = Layer.Layer(nodeNum = self._k, activation = nn.relu, grad = d_relu)
#         self._outputLayer = Layer.Layer(nodeNum = self._inputNum, activation = sig, grad = d_sig)
        self._outputLayer = Layer.Layer(nodeNum = self._inputNum)
        
        self._layers = [self._wmatrixLayer,self._outputLayer]
        
        # Connections
        # input layer to smooth Layer
        # 1-self._alpha
#         self._alpha = theano.shared(0.5)
        self._alpha = 0.5
        
        rat = np.sqrt(self._inputNum * self._k)
        # smooth layer to context layer
        # 1
        
        # context layer to smooth layer
        # W * self._alpha
        
        # smooth layer to h matrix layer
        # _h_star, _b_star
        _values = np.asarray(rng.uniform(low=0.99/rat,high=1./rat,size=(self._inputNum, self._k)))
#         self._h_star = theano.shared(value=_values, name='W_star', borrow=True)
#         self._b_star = theano.shared(np.zeros(self._k))
        self._h_star = _values
#         self._b_star = np.zeros(self._k)
        
        # h matrix layer to output layer
        # _w, _b  
        _values = np.asarray(rng.uniform(low=0.99/rat,high=1./rat,size=(self._k, self._inputNum)))
#         self._h = theano.shared(value=_values, name='W', borrow=True)
#         self._b = theano.shared(np.zeros(self._inputNum))
        self._h = _values
#         self._b = np.zeros(self._inputNum)
    
    # one epoch
    def learn(self,inputData):
        if not len(inputData[0]) == self._inputNum:
            raise ValueError("Input number error: get " + str(len(inputData[0])) + ", expect " + str(self._inputNum))
        self._miniBatchSize = len(inputData)
        
        self._lhd = self._lhc - self._hc
        self._lhc = self._hc
        self._err = 0;
        h_change = np.zeros((self._k,self._inputNum))
        h_star_change = np.zeros((self._inputNum,self._k))
        
        self._W=[]
        self._W_in=[]
        self._O=[]
        self._O_in=[]
        count = 0
        
        for sample in inputData:
            t0 = time.time()
            count += 1
            
            """ define neural network """
            """ Forward """
            # define input layer        
            x = sample
            
            # define w matrix layer
#             hmIn = np.dot(smOut,self._h_star) + self._b_star
            wmOut = self._wmatrixLayer.computeOut(np.dot(x,self._h_star))
            
            self._W.append(np.copy(wmOut))
            
            # define out layer
#             ouIn = np.dot(hmOut,self._h) + self._b
            ouIn = np.dot(wmOut,self._h)
            ouOut = self._outputLayer.computeOut(ouIn)
            
            self._O_in.append(np.copy(ouIn))
            self._O.append(np.copy(ouOut))
            
            # error function
            terr = ((x - ouOut) ** 2).sum()
            self._err += terr
            
            """ Back propagation """
            # error for out layer
            self._outputLayer.setErrors(
                (sample - self._outputLayer.getOutValues()) 
                * self._outputLayer.getGrad()(self._outputLayer.getOutValues()))
            # error for h matrix layer
            self._wmatrixLayer.setErrors(
                np.dot(self._outputLayer.getErrors(),np.transpose(self._h)) 
                * self._wmatrixLayer.getGrad()(self._wmatrixLayer.getOutValues()))
            
            """ update weights"""
            h_change += np.dot(np.transpose([self._wmatrixLayer.getOutValues()]), [self._outputLayer.getErrors()])
#             b_change += self._outputLayer.getErrors()
            h_star_change += np.dot(np.transpose([x]), [self._wmatrixLayer.getErrors()])
#             b_star_change += self._wmatrixLayer.getErrors()
#             print(time.time()-t0)
            
#             print(str(count) + "/" + str(len(inputData)) + "err:" + str(terr) + "(" + str(time.time()-t0) + "s)")
#             print("____________")
        
        """ update network"""
        
        t0 = time.time()
        
        h_change /= self._miniBatchSize
        h_star_change /= self._miniBatchSize
        
        self._d_h = np.copy(h_change)
        self._d_h_star = np.copy(h_star_change)
        
        for layer in self._layers:
            layer.update()
        
        # penalty extra err
        a = T.matrix()
        f_penalty = theano.function([a], self._lamb * nlg.trace(T.dot(S.structured_dot(a,self.L),T.transpose(a))))
        penalty = f_penalty(self._h)
        f_h_change = theano.function([a],-2 * self._lamb * T.transpose(S.structured_dot(self.L,T.transpose(a))))
        h_change2 = f_h_change(self._h)
        
        self._hpe = h_change2
        
        # update weight
        self._hc = np.sum((self._eta * h_change + h_change2)**2)
        self._h += self._eta * h_change + h_change2
        self._h_star += self._eta * h_star_change
        
        print("err=" + str(self._err / self._miniBatchSize) + " penalty=" + str(penalty))
        self._err = self._err / self._miniBatchSize  + penalty 
#         if self._eta<self._eta_max:
#             self._eta *= self._eta_increase_ratio
#             self._lamb *= self._eta_increase_ratio
        
    def train(self,inputData):
        epoch_num = 0
        while True:
            t0 = time.time()
            epoch_num += 1
            self.learn(inputData)
            print("epoch " + str(epoch_num) + ": tot_err =" + str(self._err) + "(" + str(time.time()-t0) + "s)")
            # Stopping rules
            if(epoch_num > 10000):
                self.save()
                break
            if(self._err < 0.01):
                self.save()
                break
            print(self._lhd,self._lhc - self._hc)
#             if(np.abs(self._hc) < 3e-7):
#                 self.save()
#                 break
            if((np.abs(self._hc) < 3e-7) and (self._lhd * (self._lhc - self._hc) < 0) and (np.abs(self._lhd * (self._lhc - self._hc))<1e-10)):
                self.save()
                break
            if epoch_num % 1 == 0:
                self.save("_" + str(epoch_num))
            
    def save(self,str=""):
        dmpnp("w" + str, self._W)
        dmpnp("d_h" + str, self._d_h)
        dmpnp("h" + str, self._h)
        dmpnp("hpe" + str, self._hpe)
        
def test_AE():
    SAMPLE_NUM = 1000
    
    a = []
    for i in range(SAMPLE_NUM):
        x1 = rng.rand()*0.1
        y1 = x1 ** 2
        x2 = rng.rand()*0.2 + 0.1
        y2 = rng.rand()*0.2 + 0.1
        a.append([x1,y1,x2,y2,0])
    for i in range(SAMPLE_NUM):
        x1 = rng.rand()*0.1
        y1 = rng.rand()*0.1
        x2 = rng.rand()*0.2 + 0.1
        y2 = x2 ** 3
        a.append([x1,y1,x2,y2,1])
    rng.shuffle(a)
    inputData = np.transpose(np.transpose(a)[0:4])
    y = np.transpose(np.transpose(a)[4])
    dmpnp("y",y)
    A = [[0,1,0,0],
         [1,0,0,0],
         [0,0,0,1],
         [0,0,1,0]]
    D = np.diag(np.sum(A,0))
    L = sp.csr_matrix(D - A)
    
    network = AE(4,L)
    network.train(inputData)
        
def dmpnp(str,info):
    np.savetxt(logpath + str + ".txt", info ,fmt='%0.6f')
            
if __name__ == '__main__':
    cwd = os.getcwd()
    logStart = time.strftime("%Y_%m_%d_%H_%M_%S")
    logpath = cwd + "/../log/" + logStart + "/"
    if not os.path.exists(logpath):
        os.makedirs(logpath)
        
    with open(logpath + time.strftime("%Y_%m_%d_%H_%M_%S") + ".txt","w") as tfile:
        tfile.write(time.strftime("%c") + '\n')
        
    test_AE()