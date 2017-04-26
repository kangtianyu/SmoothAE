'''
Created on Apr 16, 2017

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
import Datasets

class SmoothAE(object):
    
    def __init__(self, inputSize, learningRule = None, costFunction = None):
        """Initialize the parameters for the Smooth Auto Encoder

        :type inputData: theano.tensor.TensorType
        :param inputData: symbolic variable that describes the input of the
        architecture (one minibatch)

        """
        
        """ Hyper-parameters """
        # Sub-space dimension         
        self._k = 3
        
        # Learning rate
        self._eta = 1e-7
        
        # Regularization penalty
        self._lamb = 0
        
        # epoch num
        self._epochNum = 0
        
        """ Initalize """
        self._inputNum = inputSize
        
        # user define grad because auto generated grad not working correctly
        # grad of relu
        d_relu = lambda x: np.greater_equal(x,0)*1.
        
        # Layers
        self._smoothLayer = Layer.Layer(nodeNum = self._inputNum)
        self._contextLayer = Layer.Layer(nodeNum = self._inputNum)
        self._hmatrixLayer = Layer.Layer(nodeNum = self._k, activation = nn.relu, grad = d_relu)
        self._outputLayer = Layer.Layer(nodeNum = self._inputNum, activation = nn.relu, grad = d_relu)
        
        self._layers = [self._smoothLayer,self._contextLayer,self._hmatrixLayer,self._outputLayer]
        
        # Connections
        # input layer to smooth Layer
        # 1-self._alpha
#         self._alpha = theano.shared(0.5)
        self._alpha = 1.
        
        # smooth layer to context layer
        # 1
        
        # context layer to smooth layer
        # ADJ_MAT * self._alpha
        
        # smooth layer to h matrix layer
        # _w_star, _b_star
        _values = np.asarray(rng.uniform(low=-0.5,high=0.5,size=(self._inputNum, self._k)))
#         self._w_star = theano.shared(value=_values, name='W_star', borrow=True)
#         self._b_star = theano.shared(np.zeros(self._k))
        self._w_star = _values
        self._b_star = np.zeros(self._k)
        
        # h matrix layer to output layer
        # _w, _b  
        _values = np.asarray(rng.uniform(low=-0.5,high=0.5,size=(self._k, self._inputNum)))
#         self._w = theano.shared(value=_values, name='W', borrow=True)
#         self._b = theano.shared(np.zeros(self._inputNum))
        self._w = _values
        self._b = np.zeros(self._inputNum)
    
    # one epoch
    def learn(self,inputData):
        if not len(inputData[0]) == self._inputNum:
            raise ValueError("Input number error: get " + str(len(inputData[0])) + ", expect " + str(self._inputNum))
        self._miniBatchSize = len(inputData)
        
        self._err = 0;
        w_change = np.zeros((self._k,self._inputNum))
        b_change = np.zeros(self._inputNum)
        w_star_change = np.zeros((self._inputNum,self._k))
        b_star_change = np.zeros(self._k)
        alpha_change = 0
        
        H=[]
        count = 0
        
        for sample in inputData:
            t0 = time.time()
            count += 1
            
            """ define neural network """
            """ Forward """
            # define input layer        
            x = sample
            # define smooth layer
            # smooth function is f_(t+1) = alpha * ADJ_MAT * f_t + (1-alpha) * f_0
            smIn =  self._alpha * S.dot(T.as_tensor_variable(self._contextLayer.getAverageOut()),Datasets.ADJ_MAT) + np.multiply(x,1-self._alpha)
            smOut = self._smoothLayer.computeOut(smIn.eval())
            
            # define context layer
    #         coIn = smOut
    #         coOut = self._contextLayer.computeOut(coIn)
            
            # define h matrix layer
            hmIn = np.dot(smOut,self._w_star) + self._b_star
            hmOut = self._hmatrixLayer.computeOut(hmIn)
            
            H.append(hmOut)
            
            # define out layer
            ouIn = np.dot(hmOut,self._w) + self._b
            ouOut = self._outputLayer.computeOut(ouIn)              
        
            # error function
            self._err += ((x - ouOut) ** 2).sum()
            # update context layer
            self._contextLayer.computeOut(self._smoothLayer.getOutValues())
            
            """ Back propagation """
            # error for out layer
            self._outputLayer.setErrors(
                (sample - self._outputLayer.getOutValues()) 
                * self._outputLayer.getGrad()(self._outputLayer.getOutValues()))
            # error for h matrix layer
            self._hmatrixLayer.setErrors(
                np.dot(self._outputLayer.getErrors(),np.transpose(self._w)) 
                * self._hmatrixLayer.getGrad()(self._hmatrixLayer.getOutValues()))
            # error for smooth layer
            self._smoothLayer.setErrors(
                (np.dot(self._hmatrixLayer.getErrors(),np.transpose(self._w_star))
                + self._contextLayer.getAverageError()) / 2
                * self._smoothLayer.getGrad()(self._smoothLayer.getOutValues()))
            # error for context layer
            self._contextLayer.setErrors(
                self._alpha
                * S.dot(T.as_tensor_variable(self._smoothLayer.getErrors()),Datasets.ADJ_MAT).eval()
                * self._contextLayer.getGrad()(self._contextLayer.getOutValues()))
            
            """ update weights"""
            w_change += np.dot(np.transpose([self._hmatrixLayer.getOutValues()]), [self._outputLayer.getErrors()])
            b_change += self._outputLayer.getErrors()
            w_star_change += np.dot(np.transpose([self._smoothLayer.getOutValues()]), [self._hmatrixLayer.getErrors()])
            b_star_change += self._hmatrixLayer.getErrors()
            alpha_change_c = S.sp_sum(S.mul(np.dot(np.transpose([self._contextLayer.getAverageOut()]), [self._smoothLayer.getErrors()]),Datasets.ADJ_MAT))/Datasets.NON_ZEROS
            alpha_change_i = np.sum(self._smoothLayer.getErrors())/self._inputNum
            alpha_change += alpha_change_c - alpha_change_i
            
            Datasets.log(str(count) + "/" + str(len(inputData))  + "(" + str(time.time()-t0) + "s)")
        
        """ update network"""
        # penalty extra err   
        penalty = self._lamb * np.sum(np.trace(np.dot(np.dot(np.transpose(H),Datasets.L),H)))
        h_penalty_err = 2 * self._lamb * np.sum(np.dot(Datasets.L,H),0)
        s_penalty_err = (np.dot(h_penalty_err,np.transpose(self._w_star)) + self._contextLayer.getAverageError()) / 2 * self._smoothLayer.getGrad()(self._smoothLayer.getAverageOut())
        
        w_star_change += np.dot(np.transpose([self._smoothLayer.getAverageOut()]), [h_penalty_err])
        b_star_change += h_penalty_err
        alpha_change_c = S.sp_sum(S.mul(np.dot(np.transpose([self._contextLayer.getAverageOut()]), [s_penalty_err]),Datasets.ADJ_MAT))/Datasets.NON_ZEROS
        alpha_change_i = np.sum(s_penalty_err)/self._inputNum
        alpha_change += alpha_change_c - alpha_change_i
        # update weight
        self._w += self._eta * w_change
        self._b += self._eta * b_change
        self._w_star += self._eta * w_star_change
        self._b_star += self._eta * b_star_change
        self._alpha += self._eta * alpha_change.eval()
        self._err = (self._err + penalty) / self._miniBatchSize        
        for layer in self._layers:
            layer.update() 
    
    def train(self,inputData):
        epoch_num = 0
        while True:
            t0 = time.time()
            epoch_num += 1
            self.learn(inputData)
            # Stopping rules
            if(epoch_num > 100):
                break
            if(self._err < 0.01):
                break
            Datasets.log("epoch " + str(epoch_num) + ": err=" + str(self._err) + "(" + str(time.time()-t0) + "s)")
            
def test_SmoothAE():
    data = Datasets.samples
    network = SmoothAE(len(Datasets.labels))
    network.train(data)
            
if __name__ == '__main__':
    test_SmoothAE()