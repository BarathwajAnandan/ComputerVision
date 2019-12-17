import numpy as np
import scipy.io
from nn import *
from collections import Counter
import matplotlib.pyplot as plt

import skimage.measure as m

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-6
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()
train_lossList  = []
# Q5.1 & Q5.2
# initialize layers here
##########################
##### your code here #####
##########################


initialize_weights(1024,hidden_size,params,'layer1')
initialize_weights(hidden_size,hidden_size,params,'layer2')
initialize_weights(hidden_size,hidden_size,params,'layer3')
initialize_weights(hidden_size,1024,params,'output')



# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:

        # forward
        h1 = forward(xb, params, 'layer1',relu)
        # print(h1)
        h2 = forward(h1, params, 'layer2', relu)
        h3 = forward(h1, params, 'layer3', relu)    
        probs = forward(h1, params, 'output', sigmoid) # print('probs: ', probs)
        # loss
        # be sure to add loss and accuracy to epoch totals 
        #loss, acc = compute_loss_and_acc(yb, probs)
        # loss
        
        loss  = np.sum((probs - xb)**2)
        # be sure to add loss and accuracy to epoch totals 
        total_loss += loss
        #total_acc += acc
        
        #print(acc)
        # backward 
        delta1 = 2*(probs-xb)
        delta2 = backwards(delta1,params,'output',sigmoid_deriv)
        delta3 = backwards(delta2,params,'layer3',relu_deriv)
        delta3 = backwards(delta2,params,'layer2',relu_deriv)
        backwards(delta2,params,'layer1',relu_deriv)
        
        # apply gradient
        
        

        params['m_'+'W' + 'output'] = 0.9* params['m_'+'W' + 'output']  - learning_rate * params['grad_W' + 'output']
        params['m_'+'b' + 'output'] = 0.9* params['m_'+'b' + 'output']  - learning_rate * params['grad_b' + 'output']
        params['W' + 'output'] += params['m_'+'W' + 'output']
        params['b' + 'output'] += params['m_'+'b' + 'output']
        
        
        params['m_'+'W' + 'layer3'] = 0.9* params['m_'+'W' + 'layer3']  - learning_rate * params['grad_W' + 'layer3']
        params['m_'+'b' + 'layer3'] = 0.9* params['m_'+'b' + 'layer3']  - learning_rate * params['grad_b' + 'layer3']
        params['W' + 'layer3'] += params['m_'+'W' + 'layer3']
        params['b' + 'layer3'] += params['m_'+'b' + 'layer3']
        
        
        
        params['m_'+'W' + 'layer2'] = 0.9* params['m_'+'W' + 'layer2']  - learning_rate * params['grad_W' + 'layer2']
        params['m_'+'b' + 'layer2'] = 0.9* params['m_'+'b' + 'layer2']  - learning_rate * params['grad_b' + 'layer2']
        params['W' + 'layer2'] += params['m_'+'W' + 'layer2']
        params['b' + 'layer2'] += params['m_'+'b' + 'layer2']
        

        params['m_'+'W' + 'layer1'] = 0.9* params['m_'+'W' + 'layer1']  - learning_rate * params['grad_W' + 'layer1']
        params['m_'+'b' + 'layer1'] = 0.9* params['m_'+'b' + 'layer1']  - learning_rate * params['grad_b' + 'layer1']
        params['W' + 'layer1'] += params['m_'+'W' + 'layer1']
        params['b' + 'layer1'] += params['m_'+'b' + 'layer1'] 
        
        
        
        

        #total_acc /= len(batches)
        

    #total_loss /= len(batches)
    #total_acc /= len(batches)
    #train_accList.append(total_acc)
    train_lossList.append(total_loss)
    
    
    #FLayer1 =  forward(valid_x,params, 'layer1',sigmoid)
    #Foutput = forward(FLayer1,params,'output',softmax)
    # loss
    #probs = Foutput
    #validloss,valid_acc = compute_loss_and_acc(valid_y,probs)
    
    #validation_accList.append(valid_acc)
    # be sure to add loss and accuracy to epoch totals 
     
    # backward 
    #delta1 = probs-valid_y
    #delta2 = backwards(delta1,params,'output',linear_deriv)
    #backwards(delta2,params,'layer1',sigmoid_deriv)
        
        
        # training loop can be exactly the same as q2!
        ##########################
        ##### your code here #####
        ##########################

        
        
        
        
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        ##########################
        ##### your code here #####
        ##########################

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9
        
        
        
# =============================================================================
# plt.plot(np.arange(1,101), train_lossList  )
# plt.title("epoch vs Loss (AutoEncoder)")
# plt.xlabel("epoch")
# plt.ylabel("loss")
# =============================================================================
        
# Q5.3.1
import matplotlib.pyplot as plt
# visualize some results
##########################
##### your code here #####
##########################

x_ = []


fig, axs = plt.subplots(10, 2, sharex=True, sharey=True)
for count,i in enumerate([0,1,101,110,3501,3530,2510,2530,1601,1665]):
    x = valid_x[i].reshape(32,32).T
    axs[count,0].imshow(x)
    x_.append(x)


h1    = forward(valid_x, params, 'layer1',relu)
h2    = forward(h1, params, 'layer2', relu)
h3    = forward(h1, params, 'layer3', relu)    
reconstructed = forward(h1, params, 'output', sigmoid) 


#fig, axs = plt.subplots(10, 2, sharex=True, sharey=True)
for count,i in enumerate([0,1,101,110,3501,3530,2510,2530,1601,1665]):
    x = reconstructed[i].reshape(32,32).T
    axs[count,1].imshow(x)
    x_.append(x)




# Q5.3.2
from skimage.measure import compare_psnr as psnr
# evaluate PSNR
##########################
##### your code here #####
##########################
total_psnr = 0
for j in range(0,(valid_x.shape[0])):
    p = psnr(valid_x[i],reconstructed[i])
    
    total_psnr += p
average_psnr = total_psnr/valid_x.shape[0]




