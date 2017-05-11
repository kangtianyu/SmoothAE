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
import theano.tensor.nlinalg as nlg
import theano.sparse.basic as S
import theano.sparse
import scipy.sparse as sp
import sim_data as Datasets

# def quickMul(a,b):
#     li = np.transpose(np.nonzero(a))
#     rows = []
#     columns = []
#     data = []
#     size =  theano.sparse.basic.csm_shape(a)[0].eval()
#     for i in li:
#         row = i[0]
#         column = i[1]
#         rows.append(row)
#         columns.append(column)
#         data.append(a[row,column]*b[row,column])
#     return sp.csr_matrix((data,(rows,columns)), shape=(size, size))

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
        self._eta = 0.5e0
        self._eta_increase_ratio = 1.2
        self._eta_max = 1e-3
        
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
        self._smoothLayer = Layer.Layer(nodeNum = self._inputNum)
        self._contextLayer = Layer.Layer(nodeNum = self._inputNum)
#         self._contextLayer.setAverageOut(np.dot(T.as_tensor_variable([rng.rand(self._inputNum)]),Datasets.W).eval()[0])
        self._wmatrixLayer = Layer.Layer(nodeNum = self._k, activation = nn.relu, grad = d_relu)
#         self._outputLayer = Layer.Layer(nodeNum = self._inputNum, activation = sig, grad = d_sig)
        self._outputLayer = Layer.Layer(nodeNum = self._inputNum)
        
        self._layers = [self._smoothLayer,self._contextLayer,self._wmatrixLayer,self._outputLayer]
        
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
        _values = np.asarray(rng.uniform(low=0.1/rat,high=1./rat,size=(self._inputNum, self._k)))
#         self._h_star = theano.shared(value=_values, name='W_star', borrow=True)
#         self._b_star = theano.shared(np.zeros(self._k))
        self._h_star = _values
        self._b_star = np.zeros(self._k)
        
        # h matrix layer to output layer
        # _w, _b  
        _values = np.asarray(rng.uniform(low=0.1/rat,high=1./rat,size=(self._k, self._inputNum)))
#         self._h = theano.shared(value=_values, name='W', borrow=True)
#         self._b = theano.shared(np.zeros(self._inputNum))
        self._h = _values
        self._b = np.zeros(self._inputNum)
    
    # one epoch
    def learn(self,inputData):
        if not len(inputData[0]) == self._inputNum:
            raise ValueError("Input number error: get " + str(len(inputData[0])) + ", expect " + str(self._inputNum))
        self._miniBatchSize = len(inputData)
        
        self._err = 0;
        h_change = np.zeros((self._k,self._inputNum))
#         b_change = np.zeros(self._inputNum)
        h_star_change = np.zeros((self._inputNum,self._k))
#         b_star_change = np.zeros(self._k)
        alpha_change = 0
        
        self._W=[]
        self._W_in=[]
        self._O=[]
        self._O_in=[]
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
            smIn =  self._alpha * np.transpose(np.dot(Datasets.W,np.transpose([self._contextLayer.getAverageOut()])))[0] + np.multiply(x,1-self._alpha)
            smOut = self._smoothLayer.computeOut(smIn)
#             print(time.time()-t0)
            # define context layer
    #         coIn = smOut
    #         coOut = self._contextLayer.computeOut(coIn)
            
            # define h matrix layer
#             hmIn = np.dot(smOut,self._h_star) + self._b_star
            wmIn = np.dot(smOut,self._h_star)
            wmOut = self._wmatrixLayer.computeOut(wmIn)
            
            self._W_in.append(np.copy(wmIn))
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
            # update context layer
            self._contextLayer.computeOut(self._smoothLayer.getOutValues())
            
            """ Back propagation """
            # error for out layer
            self._outputLayer.setErrors(
                (sample - self._outputLayer.getOutValues()) 
                * self._outputLayer.getGrad()(self._outputLayer.getOutValues()))
            # error for h matrix layer
            self._wmatrixLayer.setErrors(
                np.dot(self._outputLayer.getErrors(),np.transpose(self._h)) 
                * self._wmatrixLayer.getGrad()(self._wmatrixLayer.getOutValues()))
            # error for smooth layer
            self._smoothLayer.setErrors(
                (np.dot(self._wmatrixLayer.getErrors(),np.transpose(self._h_star))
                + self._contextLayer.getAverageError()) / 2
                * self._smoothLayer.getGrad()(self._smoothLayer.getOutValues()))
            # error for context layer
#             print(time.time()-t0)
            self._contextLayer.setErrors(
                self._alpha
                * np.dot([self._smoothLayer.getErrors()],Datasets.W)[0]
                * self._contextLayer.getGrad()(self._contextLayer.getOutValues()))
#             print(time.time()-t0)
            
            """ update weights"""
            h_change += np.dot(np.transpose([self._wmatrixLayer.getOutValues()]), [self._outputLayer.getErrors()])
#             b_change += self._outputLayer.getErrors()
            h_star_change += np.dot(np.transpose([self._smoothLayer.getOutValues()]), [self._wmatrixLayer.getErrors()])
#             b_star_change += self._wmatrixLayer.getErrors()
#             print(time.time()-t0)
            alpha_change_c = np.sum(np.multiply(Datasets.W,np.dot(np.transpose([self._contextLayer.getAverageOut()]), [self._smoothLayer.getErrors()])))/Datasets.W_SUM         
            alpha_change_i = np.sum(np.multiply(x,self._smoothLayer.getErrors()))/self._inputNum
            alpha_change += alpha_change_c - alpha_change_i
#             print(time.time()-t0)
#             atmp = S.sp_sum(S.mul(Datasets.W,S.dot(S.transpose(sp.csr_matrix(np.asarray([self._contextLayer.getAverageOut()]))), sp.csr_matrix(np.asarray([self._smoothLayer.getErrors()]))))).eval()
#             alpha_change_c = atmp/Datasets.NON_ZEROS
#             alpha_change_i = np.sum(self._smoothLayer.getErrors())/self._inputNum
#             alpha_change += alpha_change_c - alpha_change_i            
#             self._d_alpha.append([np.copy(alpha_change_c),np.copy(alpha_change_i),np.copy(alpha_change)])
            self._d_alpha.append(np.copy(alpha_change))
            
#             Datasets.log(str(count) + "/" + str(len(inputData)) + "err:" + str(terr) + "(" + str(time.time()-t0) + "s)")
#             print("____________")
        
        """ update network"""
        
        t0 = time.time()
        
        h_change /= self._miniBatchSize
#         b_change /= self._miniBatchSize
        h_star_change /= self._miniBatchSize
#         b_star_change /= self._miniBatchSize
        alpha_change /= self._miniBatchSize
        
        self._d_h = np.copy(h_change)
#         self._d_b = b_change
        self._d_h_star = np.copy(h_star_change)
        
        for layer in self._layers:
            layer.update()
        
        # penalty extra err
        a = T.matrix()
        f_penalty = theano.function([a], self._lamb * nlg.trace(T.dot(S.structured_dot(a,Datasets.L),T.transpose(a))))
        penalty = f_penalty(self._h)
        f_h_change = theano.function([a],-2 * self._lamb * T.transpose(S.structured_dot(Datasets.L,T.transpose(a))))
        h_change2 = f_h_change(self._h)
        
        self._hpe = h_change2
        
        # update weight
        self._h += self._eta * h_change + h_change2
#         self._b += self._eta * b_change
        self._h_star += self._eta * h_star_change
#         self._b_star += self._eta * b_star_change
#         self._alpha += self._eta * alpha_change
        self._alpha += 0.1 * alpha_change

        if self._alpha > 0.89: self._alpha = 0.89
        if self._alpha < 0: self._alpha = 0
        
        Datasets.log("err=" + str(self._err / self._miniBatchSize) + " penalty=" + str(penalty))
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
            Datasets.log("epoch " + str(epoch_num) + ": tot_err =" + str(self._err) + " alpha = " + str(self._alpha) + "(" + str(time.time()-t0) + "s)")
            # Stopping rules
            if(epoch_num > 1000):
                self.save()
                break
            if(self._err < 0.01):
                self.save()
                break
            if epoch_num % 1 == 0:
                self.save("_" + str(epoch_num))
            
    def save(self,str=""):
        Datasets.dmpnp("w" + str, self._W)
        Datasets.dmpnp("w_in" + str, self._W_in)
        Datasets.dmpnp("o" + str, self._O)
        Datasets.dmpnp("o_in" + str, self._O_in)
        Datasets.dmpnp("h" + str, self._h)
        Datasets.dmpnp("h_star" + str, self._h_star)
#         Datasets.dmpnp("B" + str, self._b)
#         Datasets.dmpnp("B_star" + str, self._b_star)
        Datasets.dmpnp("d_alpha" + str, self._d_alpha)
        Datasets.dmpnp("d_h" + str, self._d_h)
#         Datasets.dmpnp("d_b" + str, self._d_b)
        Datasets.dmpnp("d_h_star" + str, self._d_h_star)
#         Datasets.dmpnp("d_b_star" + str, self._d_b_star)
        Datasets.dmpnp("hpe" + str, self._hpe)
        Datasets.dmpnp("s_err" + str, self._smoothLayer.getAverageError())
        Datasets.dmpnp("c_err" + str, self._contextLayer.getAverageError())
        Datasets.dmpnp("h_err" + str, self._wmatrixLayer.getAverageError())
        Datasets.dmpnp("o_err" + str, self._outputLayer.getAverageError())
        Datasets.dmpnp("s_out" + str, self._smoothLayer.getAverageOut())
        Datasets.dmpnp("c_out" + str, self._contextLayer.getAverageOut())
#         Datasets.dmpnp("h_out" + str, self._wmatrixLayer.getAverageOut())
#         Datasets.dmpnp("o_out" + str, self._outputLayer.getAverageOut())
        
def test_SmoothAE():
    data = Datasets.samples
    network = SmoothAE(len(Datasets.labels))
    network.train(data)
            
if __name__ == '__main__':
    test_SmoothAE()