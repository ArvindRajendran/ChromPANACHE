# -*- coding: utf-8 -*-

#This code was built on the original code written by Mazziar Raissi: https://github.com/maziarraissi/PINNs

#author: Gokul Subraveti

import tensorflow as tf
import numpy as np
import scipy.io
from scipy.interpolate import griddata
import time


np.random.seed(1234)
tf.set_random_seed(1234)

class PANACHE:
    # Initialize the class
    def __init__(self, X0, X_lb, X_rb, in_train, lb_train, rb_train, 
                 X_c, layers, low_bound, up_bound, coeff1, coeff2, coeff3, N0, N_b, N_c):
        
        self.low_bound = low_bound
        self.up_bound  = up_bound

        self.N0     = N0
        self.N_b    = N_b 
        self.N_c    = N_c
        
        self.z0     = X0[:,0:1]
        self.t0     = X0[:,1:2]
        self.w0     = X0[:,2:3]

        self.z_lb   = X_lb[:,0:1]
        self.t_lb   = X_lb[:,1:2]
        self.w_lb   = X_lb[:,2:3]

        self.z_rb   = X_rb[:,0:1]
        self.t_rb   = X_rb[:,1:2]
        self.w_rb   = X_rb[:,2:3]
                
        self.z_c    = X_c[:,0:1]
        self.t_c    = X_c[:,1:2]
        self.w_c    = X_c[:,2:3]
        
        self.c0     = in_train[:,0:1]
        self.c_lb   = lb_train[:,0:1]
        self.c_rb   = rb_train[:,0:1]
       
        self.layers = layers
        self.coeff1 = coeff1
        self.coeff2 = coeff2
        self.coeff3 = coeff3
        
        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)

        #Assign tensorflow placeholders and session
        self.sess    = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        self.z0_tf   = tf.placeholder(tf.float32, shape=[None, self.z0.shape[1]])
        self.t0_tf   = tf.placeholder(tf.float32, shape=[None, self.t0.shape[1]]) 
        self.w0_tf   = tf.placeholder(tf.float32, shape=[None, self.w0.shape[1]])

        self.z_lb_tf = tf.placeholder(tf.float32, shape=[None, self.z_lb.shape[1]])
        self.t_lb_tf = tf.placeholder(tf.float32, shape=[None, self.t_lb.shape[1]]) 
        self.w_lb_tf = tf.placeholder(tf.float32, shape=[None, self.w_lb.shape[1]]) 

        self.z_rb_tf = tf.placeholder(tf.float32, shape=[None, self.z_rb.shape[1]])
        self.t_rb_tf = tf.placeholder(tf.float32, shape=[None, self.t_rb.shape[1]]) 
        self.w_rb_tf = tf.placeholder(tf.float32, shape=[None, self.w_rb.shape[1]]) 
        
        self.c0_tf   = tf.placeholder(tf.float32, shape=[None, self.c0.shape[1]])
        self.c_lb_tf = tf.placeholder(tf.float32, shape=[None, self.c_lb.shape[1]])
        self.c_rb_tf = tf.placeholder(tf.float32, shape=[None, self.c_rb.shape[1]])
        
        self.z_c_tf  = tf.placeholder(tf.float32, shape=[None, self.z_c.shape[1]])
        self.t_c_tf  = tf.placeholder(tf.float32, shape=[None, self.t_c.shape[1]])   
        self.w_c_tf  = tf.placeholder(tf.float32, shape=[None, self.w_c.shape[1]]) 
      
        self.N0      = tf.cast(self.N0, dtype=tf.int32)
        self.N_b     = tf.cast(self.N_b, dtype=tf.int32)
        self.N_c     = tf.cast(self.N_c, dtype=tf.int32)

        
        self.c0_pred, self.q0_pred  = self.net_chrom(self.z0_tf, self.t0_tf, self.w0_tf) 
        self.c_lb_pred, _           = self.net_chrom(self.z_lb_tf, self.t_lb_tf, self.w_lb_tf) 
        self.c_rb_pred, _           = self.net_chrom(self.z_rb_tf, self.t_rb_tf, self.w_rb_tf) 
        self.f_c_pred               = self.net_f(self.z_c_tf, self.t_c_tf, self.w_c_tf) 
      
        self.loss = tf.reduce_sum(tf.math.unsorted_segment_mean(tf.square(self.c0_tf - self.c0_pred), self.N0, 3)) + \
                    tf.reduce_sum(tf.math.unsorted_segment_mean(tf.square(self.q0_pred), self.N0, 3)) + \
                    tf.reduce_sum(tf.math.unsorted_segment_mean(tf.square(self.c_lb_tf - self.c_lb_pred), self.N_b, 3)) + \
                    tf.reduce_sum(tf.math.unsorted_segment_mean(tf.square(self.c_rb_tf - self.c_rb_pred), self.N_b, 3)) + \
                    tf.reduce_sum(tf.math.unsorted_segment_mean(tf.square(self.f_c_pred), self.N_c, 3))
        
                
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.saver = tf.train.Saver()
            
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.random.normal([1,layers[l+1]], 0, 1, dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)       
        return weights, biases
      
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]      
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.low_bound)/(self.up_bound - self.low_bound) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b, name="Y_out")
        return Y

    def net_chrom(self, z, t, w):
        chrom = self.neural_net(tf.concat([z,t,w],1), self.weights, self.biases)
        
        c = chrom[:,0:1]
        q = chrom[:,1:2]
        return c, q
      
    def net_f(self, z,t,w):
        c, q = self.net_chrom(z,t,w)
        c_t = tf.gradients(c, t)[0]
        q_t = tf.gradients(q, t)[0]
        c_z = tf.gradients(c, z)[0]
        c_zz = tf.gradients(c_z, z)[0]

        f_c = c_t + self.coeff1*c_z - self.coeff2*c_zz + self.coeff3*q_t
        return f_c
      
    def callback(self, loss):
        print('Loss:', loss)
        
    def train(self):
        
        tf_dict = {self.z0_tf: self.z0, self.t0_tf: self.t0, self.w0_tf: self.w0, 
                   self.z_lb_tf: self.z_lb, self.t_lb_tf: self.t_lb, self.w_lb_tf: self.w_lb,  
                   self.z_rb_tf: self.z_rb, self.t_rb_tf: self.t_rb, self.w_rb_tf: self.w_rb, 
                   self.c0_tf: self.c0, self.c_lb_tf: self.c_lb, self.c_rb_tf: self.c_rb, 
                   self.z_c_tf: self.z_c, self.t_c_tf: self.t_c, self.w_c_tf: self.w_c}
                                                                                                                          
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback) 
    def save(self):
        saved_model = self.saver.save(self.sess, 'model')
        return saved_model
           
    def predict(self, X_hat):
                
        c_hat   = self.sess.run(self.c_rb_pred, {self.z_rb_tf: X_hat[:,0:1], self.t_rb_tf: X_hat[:,1:2], self.w_rb_tf: X_hat[:,2:3]}) 
        f_c_hat = self.sess.run(self.f_c_pred, {self.z_c_tf: X_hat[:,0:1], self.t_c_tf: X_hat[:,1:2], self.w_c_tf: X_hat[:,2:3]})
        w_hat   = self.sess.run(self.weights, {self.z_rb_tf: X_hat[:,0:1], self.t_rb_tf: X_hat[:,1:2], self.w_rb_tf: X_hat[:,2:3]})
        b_hat   = self.sess.run(self.biases, {self.z_rb_tf: X_hat[:,0:1], self.t_rb_tf: X_hat[:,1:2], self.w_rb_tf: X_hat[:,2:3]})
               
        return c_hat, f_c_hat, w_hat, b_hat

if __name__ == "__main__": 
    
    #Load data from MATLAB
    data      = scipy.io.loadmat('train_chrom.mat')
    Z         = np.real(data['Z'])
    T         = np.real(data['T']) 
    X0        = np.real(data['X0'])
    X_lb      = np.real(data['X_lb'])
    X_rb      = np.real(data['X_rb'])
    X_c_train = np.real(data['X_c_train'])
    X_sol_1   = np.real(data['X_sol_1'])
    in_train  = np.real(data['in_train'])
    lb_train  = np.real(data['lb_train'])
    rb_train  = np.real(data['rb_train'])  
    low_bound = np.real(data['low_bound'])
    up_bound  = np.real(data['up_bound'])
    N0_ids    = np.real(data['N0_ids'])
    N_b_ids   = np.real(data['N_b_ids'])
    N_c_ids   = np.real(data['N_c_ids'])
    
    #Define constants
    ep_b    = 0.648
    v0      = 1.964875841      #cm/min
    tfinal  = 21               #min
    length  = 10               #cm
    ax_disp = 0.1071           #cm2/min single  
    phi     = (1-ep_b)/(ep_b)
    
    #Define PDE coefficients
    coeff1  = v0*tfinal/(length)
    coeff2  = ax_disp*tfinal/(length ** 2)
    coeff3  = phi     
    
    #Neural network architecture
    layers  = [3, 50, 50, 50, 50, 50,  2]
    
    #Index IDs
    N0_ids  = N0_ids.reshape(-1)
    N_b_ids = N_b_ids.reshape(-1)
    N_c_ids = N_c_ids.reshape(-1)

    
    model = PANACHE(X0, X_lb, X_rb, in_train, lb_train, rb_train,
                              X_c_train, layers, low_bound, up_bound, 
                    coeff1, coeff2, coeff3, N0_ids, N_b_ids, N_c_ids)
    
    #Train the model
    start_time = time.time()                
    model.train()
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))
    
    #Save the model
    model.save()


    c_hat, f_c_hat, w_hat, b_hat = model.predict(X_sol_1)
            
    
    bias = np.empty((len(b_hat),), dtype=np.object)
    for i in range(len(b_hat)):
        bias[i] = b_hat[i]

    
    scipy.io.savemat('neuralnetparameters.mat',{"bias":bias, 'w_hat': w_hat})



