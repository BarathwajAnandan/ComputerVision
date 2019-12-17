import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None

    ##########################
    ##### your code here #####
    ##########################
    #W = 2/(in_size+out_size)
    W = np.random.uniform(-np.sqrt(6)/np.sqrt(out_size + in_size) ,np.sqrt(6)/np.sqrt(out_size + in_size),(in_size,out_size))
    
    
    b = np.zeros((out_size))
    #print(b.shape)
    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    
    
    
    
    res = None


    res = 1/ (1+np.exp(-x))


    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]
    
    
    #print(W.shape)
#pre and post in a loop or will it be called in the main ?
    pre_act = np.dot(X,W) + b
    
    post_act = activation(pre_act)
    

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    # print('x: ', x)

    res = np.zeros(x.shape)
    c = -np.max(x, axis=1).reshape((x.shape[0],1))
    # print("x shape: ", x.shape)
    # print("c shape: ", c.shape)
    
    x += c
    ex = np.exp(x)
    deno_Sum = np.sum(ex, axis=1)
    res = np.divide(ex, deno_Sum.reshape((-1,1)))    
    #for i in range(x.shape[0]):
     #   res[i] = np.exp(x[i]) / np.sum(x)
    
    

    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None

    loss = - np.sum(y * np.log(probs))
    
    correctPredictions = np.sum(np.equal(np.argmax(y, axis=-1), np.argmax(probs, axis=-1)))
    totalPredictions = y.shape[0]
    acc = correctPredictions/totalPredictions


    return loss, acc 

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # do the derivative through activation first
    # then compute the derivative W,b, and X
    
    act_der_del = activation_deriv(post_act)  #finding the der of loss Func
    delta = delta * act_der_del

    grad_W = np.dot(X.T, delta) 
    grad_X = np.dot(delta, W.T)
    
    
    #print("111", X.shape,W.shape,b.shape, delta.shape)
    
    y_wrt_b = np.ones((1, delta.shape[0]))
    
    grad_b = np.dot(y_wrt_b, delta).reshape(-1)
    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    
    
  #  batchx = np.zeros((batch_size, x.shape[1]))
  #  batchy = np.zeros((batch_size, y.shape[1]))
    
    
    for i in range(0,np.int(x.shape[0]/batch_size)):
        
        idx = np.random.choice(np.arange(x.shape[0]), size=batch_size, replace=False)   
        batchx = x[idx]
        batchy = y[idx]
        batches.append((batchx,batchy))

    return batches
