

# Project name : Implementation of SVM Classifier


import numpy as np
import matplotlib.pyplot as plt
import SVM_classifier as svm

## -------------------------------------------------- ## Loading data
# Loading function
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Load train data and combine them
batch1 = unpickle("data_batch_1")
Xtr1 = batch1[b'data']
Ytr1 = np.array(batch1[b'labels'])
batch2 = unpickle("data_batch_2")
Xtr2 = batch2[b'data']
Ytr2 = np.array(batch2[b'labels'])
batch3 = unpickle("data_batch_3")
Xtr3 = batch3[b'data']
Ytr3 = np.array(batch3[b'labels'])
batch4 = unpickle("data_batch_4")
Xtr4 = batch4[b'data']
Ytr4 = np.array(batch4[b'labels'])
batch5 = unpickle("data_batch_5")
Xtr5 = batch5[b'data']
Ytr5 = np.array(batch5[b'labels'])

Xtr_f = np.append(np.append(np.append(Xtr1, Xtr2, axis=0), np.append(Xtr3, Xtr4, axis=0), axis=0), Xtr5, axis=0)
Ytr_f = np.append(np.append(np.append(Ytr1, Ytr2, axis=0), np.append(Ytr3, Ytr4, axis=0), axis=0), Ytr5, axis=0)

# Load test data
testbatch = unpickle("test_batch")
Xte = testbatch[b'data']
Yte = np.array(testbatch[b'labels'])

# Bias trick
Xtr = np.append(Xtr_f, np.ones((Xtr_f.shape[0], 1)), axis = 1)
Xtr_test = np.append(Xte, np.ones((Xte.shape[0], 1)), axis = 1)


## -------------------------------------------------- ## Optimization
# Initialization
# scale down the size of W by multiplying small number
W1 = np.random.randn(3073, 10) * 0.0001
W2 = np.random.randn(3073, 10) * 0.0001
n1 = 0
n2 = 0
x1=list()
x2=list()
y1=list()
y2=list()
y3=list()
y4=list()

# Set hyperparameter
iteration1 = 10000
iteration2 = 10000
step_size = 0.00000001
sampling_num = 256

# Gradient Descent
while n1 <= iteration1:
    # return loss and dW
    [loss, dW] = svm.L_i_vectorized(Xtr, Ytr_f, W1)

    # calculate accuracy
    predict = np.argmax(Xtr_test.dot(W1), axis=1)
    acc = np.mean(predict == Yte)

    # print result every 100 iteration
    if n1 % 100 == 0:
        print('Iteration >> %i' % (n1))
        print('     Loss: %f' % (loss))
        print('     Accuracy : %f' % (np.mean(predict == Yte)))

    # update W and n
    W1 += -step_size * dW
    n1 += 1

    # save result to draw graph
    x1.append(n1)
    y1.append(loss)
    y2.append(acc)

# Mini-batch Gradient Descent
while n2 <= iteration2:
    # select random sample
    batch_index = np.random.choice(Xtr.shape[1], sampling_num)

    # return loss and dW
    [loss, dW] = svm.L_i_vectorized(Xtr[batch_index], Ytr_f[batch_index], W2)


    # calculate accuracy
    predict = np.argmax(Xtr_test.dot(W2), axis=1)
    acc = np.mean(predict == Yte)

    # print result every 100 iteration
    if n2 % 100 == 0:
        print('Iteration >> %i' % (n2))
        print('     Loss: %f' % (loss))
        print('     Accuracy : %f' % (np.mean(predict == Yte)))

    # update W and n
    W2 += -step_size * dW
    n2 += 1

    # save result to draw graph
    x2.append(n2)
    y3.append(loss)
    y4.append(acc)


## -------------------------------------------------- ## Visualization Graph
# Strings
xlab = 'Iteration'
ylab1 = 'Loss'
ylab2 = 'Acc'
title1 = 'Loss per Iteration'
title2 = 'Acc per Iteration'

# Gradient Descent Graph
# Plotting loss per iteration
plt.figure()
plt.subplot(2,2,1)
plt.plot(x1,y1)
plt.xlabel(xlab)
plt.ylabel(ylab1)
plt.title(title1)

# Plotting accuracy per iteration
plt.subplot(2,2,2)
plt.plot(x1,y2)
plt.xlabel(xlab)
plt.ylabel(ylab2)
plt.title(title2)

# Mini-batch Gradient Descent Graph
# Plotting loss per iteration
plt.subplot(2,2,3)
plt.plot(x2,y3)
plt.xlabel(xlab)
plt.ylabel(ylab1)
plt.title(title1)

# Plotting accuracy per iteration
plt.subplot(2,2,4)
plt.plot(x2,y4)
plt.xlabel(xlab)
plt.ylabel(ylab2)
plt.title(title2)

plt.show()


## -------------------------------------------------- ## Visualization W1
# Remove the bias and reshape
W_v = W1[:-1, :].reshape(32, 32, 3, 10)

# Change scale of W1 linearly and showing image
plt.subplot(2, 5, 1)
plt.imshow(np.int8((W_v[:, :, :, 0] - np.min(W_v)) / (np.max(W_v) - np.min(W_v)) * 255.0))
plt.title('Class1')
plt.axis('off')

plt.subplot(2, 5, 2)
plt.imshow(np.int8((W_v[:, :, :, 1] - np.min(W_v)) / (np.max(W_v) - np.min(W_v)) * 255.0))
plt.title('Class2')
plt.axis('off')

plt.subplot(2, 5, 3)
plt.imshow(np.int8((W_v[:, :, :, 2] - np.min(W_v)) / (np.max(W_v) - np.min(W_v)) * 255.0))
plt.title('Class3')
plt.axis('off')

plt.subplot(2, 5, 4)
plt.imshow(np.int8((W_v[:, :, :, 3] - np.min(W_v)) / (np.max(W_v) - np.min(W_v)) * 255.0))
plt.title('Class4')
plt.axis('off')

plt.subplot(2, 5, 5)
plt.imshow(np.int8((W_v[:, :, :, 4] - np.min(W_v)) / (np.max(W_v) - np.min(W_v)) * 255.0))
plt.title('Class5')
plt.axis('off')

plt.subplot(2, 5, 6)
plt.imshow(np.int8((W_v[:, :, :, 5] - np.min(W_v)) / (np.max(W_v) - np.min(W_v)) * 255.0))
plt.title('Class6')
plt.axis('off')

plt.subplot(2, 5, 7)
plt.imshow(np.int8((W_v[:, :, :, 6] - np.min(W_v)) / (np.max(W_v) - np.min(W_v)) * 255.0))
plt.title('Class7')
plt.axis('off')

plt.subplot(2, 5, 8)
plt.imshow(np.int8((W_v[:, :, :, 7] - np.min(W_v)) / (np.max(W_v) - np.min(W_v)) * 255.0))
plt.title('Class8')
plt.axis('off')

plt.subplot(2, 5, 9)
plt.imshow(np.int8((W_v[:, :, :, 8] - np.min(W_v)) / (np.max(W_v) - np.min(W_v)) * 255.0))
plt.title('Class9')
plt.axis('off')

plt.subplot(2, 5, 10)
plt.imshow(np.int8((W_v[:, :, :, 9] - np.min(W_v)) / (np.max(W_v) - np.min(W_v)) * 255.0))
plt.title('Class10')
plt.axis('off')

plt.show()
