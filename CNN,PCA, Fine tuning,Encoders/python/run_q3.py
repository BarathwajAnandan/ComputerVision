import numpy as np
import scipy.io
from nn import *

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

max_iters = 100
# pick a batch size, learning rate
batch_size = 32


learning_rate = 0.003
hidden_size = 64
##########################
##### your code here #####
##########################

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
##########################
##### your code here #####
##########################
initialize_weights(1024,hidden_size,params,'layer1')
initialize_weights(hidden_size,36,params,'output')




train_accList = []
train_lossList = []
validation_accList  = []


# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    valid_acc = None
    for xb,yb in batches:
        # forward
        h1 = forward(xb, params, 'layer1')
        # print(h1)
        probs = forward(h1, params, 'output', softmax)
        # print('probs: ', probs)
        # loss
        # be sure to add loss and accuracy to epoch totals 
        loss, acc = compute_loss_and_acc(yb, probs)
        # loss
        
        loss,acc = compute_loss_and_acc(yb,probs)
        # be sure to add loss and accuracy to epoch totals 
        total_loss += loss
        total_acc += acc
        
        #print(acc)
        # backward 
        delta1 = probs-yb
        delta2 = backwards(delta1,params,'output',linear_deriv)
        backwards(delta2,params,'layer1',sigmoid_deriv)
        # apply gradient

        params['W' + 'output'] -= learning_rate * params['grad_W' + 'output']
        params['b' + 'output'] -= learning_rate * params['grad_b' + 'output']
        params['W' + 'layer1'] -= learning_rate * params['grad_W' + 'layer1']
        params['b' + 'layer1'] -= learning_rate * params['grad_b' + 'layer1']
        #total_acc /= len(batches)
        

    total_loss /= len(batches)
    total_acc /= len(batches)
    train_accList.append(total_acc)
    train_lossList.append(total_loss)
    FLayer1 =  forward(valid_x,params, 'layer1',sigmoid)
    Foutput = forward(FLayer1,params,'output',softmax)
    # loss
    probs = Foutput
    validloss,valid_acc = compute_loss_and_acc(valid_y,probs)
    
    validation_accList.append(valid_acc)
    # be sure to add loss and accuracy to epoch totals 
     
    # backward 
    delta1 = probs-valid_y
    delta2 = backwards(delta1,params,'output',linear_deriv)
    backwards(delta2,params,'layer1',sigmoid_deriv)
        
        
        # training loop can be exactly the same as q2!
        ##########################
        ##### your code here #####
        ##########################

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))
        print(" validation :\t acc : {:.2f}".format(valid_acc))
# run on validation set and report accuracy! should be above 75%
#avg_acc /= len(batches)


#learned weight plot.
        




#for plotting the Accuracy and loss graphs
# =============================================================================
# import matplotlib.pyplot as plt
# plt.figure(1)
# plt.plot(np.arange(1,101), train_accList )
# plt.plot(np.arange(1,101), validation_accList )
# plt.gca().legend(('train_accuracy', 'validation_accuracy'))
# plt.title("epoch vs accuracy (learningrate*10 (0.03))")
# plt.xlabel("epoch")
# plt.ylabel("accuracy")
# 
# #plt.figure(2)
# #plt.plot(np.arange(1,101), validation_accList )
# 
# 
# 
# 
# 
# 
# plt.figure(3)
# plt.plot(np.arange(1,101), train_lossList )
# plt.title("trainData_loss (learningRate*10 (0.03)) ")
# plt.xlabel("epoch")
# plt.ylabel("loss")
# 
# 
# =============================================================================

# =============================================================================
# plt.plot(np.arange(1,101), )
# plt.title("epoch vs Loss (learningrate*10 (0.03))")
# plt.xlabel("epoch")
# plt.ylabel("accuracy")
# 
# 
# #test :
# 
# 
# FLayer1 =  forward(test_x,params, 'layer1',sigmoid)
# probs = forward(FLayer1,params,'output',softmax)
# # loss
# validloss,test_acc = compute_loss_and_acc(test_y,probs)
# # be sure to add loss and accuracy to epoch totals 
#  
# # backward 
# delta1 = probs-test_y
# delta2 = backwards(delta1,params,'output',linear_deriv)
# backwards(delta2,params,'layer1',sigmoid_deriv)
# 
# print(probs)
# print(" Test :\t acc : {:.2f}".format(test_acc))
# 
# =============================================================================




# training loop can be exactly the same as q2!
##########################
##### your code here #####
##########################

#if itr % 2 == 0:
#print(" validation :\t acc : {:.2f}".format(valid_acc))



##########################
##### your code here #####
##########################

#print('Validation accuracy: ',valid_acc)
if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
        
        
        # for the weight plot - initial and final!
# =============================================================================
# import pickle
# saved_params = {k:v for k,v in params.items() if '_' not in k}
# with open('q3_weights.pickle', 'wb') as handle:
#     pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
# 
# # Q3.1.3
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import ImageGrid
# 
# 
# with open('q3_weights.pickle', 'rb') as handle:
#    saved_params = pickle.load(handle)
#    
# weights = saved_params['Wlayer1']
#    
# fig = plt.figure(7)
# grid = ImageGrid(fig, 111, (8,8))
# 
# for i in range(weights.shape[1]):
#     print(i)
#     im = grid[i].imshow(weights[:,i].reshape(32,-1))
# plt.show()
# 
# 
# 
# 
# fig = plt.figure(8)
# grid = ImageGrid(fig, 111, (8,8))
# initialize_weights(1024,hidden_size,saved_params,'initialWeights')
# iniweights = saved_params['WinitialWeights']
# for i in range(iniweights.shape[1]):
#     print(i)
#     im = grid[i].imshow(iniweights[:,i].reshape(32,-1))
# 
# plt.show()
# 
# 
# =============================================================================











# Q3.1.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

# compute comfusion matrix here
##########################
##### your code here #####
##########################

label = np.argmax(test_y,axis = 1)
predicted = np.argmax(probs,axis =1)

for i in range(1800):
    
    confusion_matrix[label[i],predicted[i]] += 1
        
        
#confusion_matrix = confusion_matrix/1800


import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()
