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
import scipy.sparse as sp
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
        self._eta = 1e-1
        
        # Regularization penalty
        self._lamb = 1e0
        
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
        self._smoothLayer = Layer.Layer(nodeNum = self._inputNum)
        self._contextLayer = Layer.Layer(nodeNum = self._inputNum)
        self._contextLayer.setAverageOut(rng.rand(self._inputNum))
        self._hmatrixLayer = Layer.Layer(nodeNum = self._k, activation = nn.relu, grad = d_relu)
        self._outputLayer = Layer.Layer(nodeNum = self._inputNum, activation = sig, grad = d_sig)
        
        self._layers = [self._smoothLayer,self._contextLayer,self._hmatrixLayer,self._outputLayer]
        
        # Connections
        # input layer to smooth Layer
        # 1-self._alpha
#         self._alpha = theano.shared(0.5)
        self._alpha = 1.
        
        # smooth layer to context layer
        # 1
        
        # context layer to smooth layer
        # W * self._alpha
        
        # smooth layer to h matrix layer
        # _w_star, _b_star
        _values = np.asarray(rng.uniform(low=0.0,high=1.0,size=(self._inputNum, self._k)))
#         self._w_star = theano.shared(value=_values, name='W_star', borrow=True)
#         self._b_star = theano.shared(np.zeros(self._k))
        self._w_star = _values
        self._b_star = np.zeros(self._k)
        
        # h matrix layer to output layer
        # _w, _b  
        _values = np.asarray(rng.uniform(low=0.0,high=1.0,size=(self._k, self._inputNum)))
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
        
        self._H=[]
        self._H_in=[]
        self._d_alpha=[]
        count = 0
        
        for sample in inputData:
            t0 = time.time()
            count += 1
            
            """ define neural network """
            """ Forward """
            # define input layer        
            x = sample
            # define smooth layer
            # smooth function is f_(t+1) = alpha * W * f_t + (1-alpha) * f_0
            smIn =  self._alpha * S.dot(T.as_tensor_variable(self._contextLayer.getAverageOut()),Datasets.W) + np.multiply(x,1-self._alpha)
            smOut = self._smoothLayer.computeOut(smIn.eval())
            
            # define context layer
    #         coIn = smOut
    #         coOut = self._contextLayer.computeOut(coIn)
            
            # define h matrix layer
            hmIn = np.dot(smOut,self._w_star) + self._b_star
            hmOut = self._hmatrixLayer.computeOut(hmIn)
            
            self._H_in.append(hmIn)
            self._H.append(hmOut)
            
            # define out layer
            ouIn = np.dot(hmOut,self._w) + self._b
            ouOut = self._outputLayer.computeOut(ouIn)
            
            # error function
            terr = ((x - ouOut) ** 2).sum()
            self._err += terr
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
                * S.dot(T.as_tensor_variable(self._smoothLayer.getErrors()),Datasets.W).eval()
                * self._contextLayer.getGrad()(self._contextLayer.getOutValues()))
            
            """ update weights"""
            w_change += np.dot(np.transpose([self._hmatrixLayer.getOutValues()]), [self._outputLayer.getErrors()])
            b_change += self._outputLayer.getErrors()
            w_star_change += np.dot(np.transpose([self._smoothLayer.getOutValues()]), [self._hmatrixLayer.getErrors()])
            b_star_change += self._hmatrixLayer.getErrors()
            alpha_change_c = S.sp_sum(S.mul(Datasets.W,S.dot(S.transpose(sp.csr_matrix(np.asarray([self._contextLayer.getAverageOut()]))), sp.csr_matrix(np.asarray([self._smoothLayer.getErrors()])))))/Datasets.NON_ZEROS
            alpha_change_i = np.sum(self._smoothLayer.getErrors())/self._inputNum
            alpha_change = (alpha_change + alpha_change_c - alpha_change_i).eval()
            
            self._d_alpha.append(alpha_change)
            
            Datasets.log(str(count) + "/" + str(len(inputData)) + "err:" + str(terr) + "(" + str(time.time()-t0) + "s)")
        
        """ update network"""
        
        t0 = time.time()
        
        w_change /= self._miniBatchSize
        b_change /= self._miniBatchSize
        w_star_change /= self._miniBatchSize
        b_star_change /= self._miniBatchSize
        alpha_change /= self._miniBatchSize
        
        # penalty extra err   
        penalty = self._lamb * np.sum(np.trace(np.dot(np.dot(np.transpose(self._H),Datasets.L),self._H)))
        h_penalty_err = 2 * self._lamb * np.sum(np.dot(Datasets.L,self._H),0)
        s_penalty_err = (np.dot(h_penalty_err,np.transpose(self._w_star)) + self._contextLayer.getAverageError()) / 2 * self._smoothLayer.getGrad()(self._smoothLayer.getAverageOut())
        
        w_star_change += np.dot(np.transpose([self._smoothLayer.getAverageOut()]), [h_penalty_err])
        b_star_change += h_penalty_err
        alpha_change_c = S.sp_sum(S.mul(S.dot(S.transpose(sp.csr_matrix(np.asarray([self._contextLayer.getAverageOut()]))), sp.csr_matrix(np.asarray([s_penalty_err]))),Datasets.W))/Datasets.NON_ZEROS
        alpha_change_i = np.sum(s_penalty_err)/self._inputNum
        alpha_change = (alpha_change + alpha_change_c - alpha_change_i).eval()
        # update weight
        self._w += self._eta * w_change 
        self._b += self._eta * b_change
        self._w_star += self._eta * w_star_change
        self._b_star += self._eta * b_star_change
        self._alpha += self._eta * alpha_change
        if self._alpha > 1: self._alpha = 1
        if self._alpha < 0: self._alpha = 0
        
        Datasets.log("err=" + str(self._err / self._miniBatchSize) + " penalty=" + str(penalty / self._miniBatchSize))
        self._err = (self._err + penalty) / self._miniBatchSize  
        for layer in self._layers:
            layer.update()
        
    def train(self,inputData):
        epoch_num = 0
        while True:
            t0 = time.time()
            epoch_num += 1
            self.learn(inputData)
            Datasets.log("epoch " + str(epoch_num) + ": tot_err =" + str(self._err) + " alpha = " + str(self._alpha) + "(" + str(time.time()-t0) + "s)")
            # Stopping rules
            if(epoch_num > 500):
                self.save()
                break
            if(self._err < 0.01):
                self.save()
                break
            if epoch_num % 1 == 0:
                self.save("_" + str(epoch_num))
            
    def save(self,str=""):
        Datasets.dmpnp("H" + str, self._H)
        Datasets.dmpnp("H_in" + str, self._H_in)
        Datasets.dmpnp("W" + str, self._w)
        Datasets.dmpnp("W_star" + str, self._w_star)
        Datasets.dmpnp("B" + str, self._b)
        Datasets.dmpnp("d_alpha" + str, self._d_alpha)
        Datasets.dmpnp("s_err" + str, self._smoothLayer.getAverageError())
        Datasets.dmpnp("c_err" + str, self._contextLayer.getAverageError())
        Datasets.dmpnp("h_err" + str, self._hmatrixLayer.getAverageError())
        Datasets.dmpnp("o_err" + str, self._outputLayer.getAverageError())
        Datasets.dmpnp("s_out" + str, self._smoothLayer.getAverageOut())
        Datasets.dmpnp("c_out" + str, self._contextLayer.getAverageOut())
        Datasets.dmpnp("h_out" + str, self._hmatrixLayer.getAverageOut())
        Datasets.dmpnp("o_out" + str, self._outputLayer.getAverageOut())
        
def test_SmoothAE():
    data = Datasets.samples
    network = SmoothAE(len(Datasets.labels))
    network.train(data)
            
if __name__ == '__main__':
    test_SmoothAE()