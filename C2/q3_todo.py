#! /usr/bin/env python3
import numpy as np
import struct
import matplotlib.pyplot as plt
from scipy.special import expit
import scipy
import scipy.sparse

def readMNISTdata():

    with open('t10k-images.idx3-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        test_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_data = test_data.reshape((size, nrows*ncols))
    
    with open('t10k-labels.idx1-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        test_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_labels = test_labels.reshape((size,1))
    
    with open('train-images.idx3-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        train_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_data = train_data.reshape((size, nrows*ncols))
    
    with open('train-labels.idx1-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        train_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_labels = train_labels.reshape((size,1))

    # augmenting a constant feature of 1 (absorbing the bias term)
    train_data = np.concatenate( ( np.ones([train_data.shape[0],1]), train_data ), axis=1)
    test_data  = np.concatenate( ( np.ones([test_data.shape[0],1]),  test_data ), axis=1)
    np.random.seed(314)
    np.random.shuffle(train_labels)
    np.random.seed(314)
    np.random.shuffle(train_data)

    X_train = train_data[:50000] / 256
    t_train = train_labels[:50000]

    X_val   = train_data[50000:] /256
    t_val   = train_labels[50000:]

    return X_train, t_train, X_val, t_val, test_data, test_labels

def predict(X, W, t = None):
    y = softmax(X @ W)
    t_hat = np.argmax(y, axis=1) 
    new_t = t[:,0]
    t1 = scipy.sparse.csr_matrix((np.ones( new_t.shape[0]), (new_t, np.array(range( new_t.shape[0])))))
    loss = -1/len(t) * np.sum(np.array(t1.todense()).T)
    acc = np.sum(t_hat == t.flatten()) / len(t_hat)
    return y, t_hat, acc,loss


def train(X_train, y_train, X_val, t_val):
    N_train = X_train.shape[0]
    N_val   = X_val.shape[0]
    w = np.zeros([X_train.shape[1], N_class])
    train_losses = []
    valid_accs = []
    W_best = None
    epoch_best = 0
    acc_best = 0
    #from assignment 1
    print('training starts...')
    for epoch in range(MaxEpoch):
        loss_this_epoch = 0
        for b in range(int(np.ceil(N_train/batch_size)) ):
            X_batch = X_train[b*batch_size : (b+1)*batch_size]
            y_batch = y_train[b*batch_size : (b+1)*batch_size]

            length = X_batch.shape[0]
            y = y_batch[:,0]
            t = scipy.sparse.csr_matrix((np.ones( y.shape[0]), (y, np.array(range( y.shape[0])))))
            t_indication = np.array(t.todense()).T
            y_hat = softmax(X_batch @ w)
            loss_batch = -1/length * np.sum(t_indication * np.log(y_hat))
            g = 1/length * X_batch.T @ (y_hat-t_indication)
            loss_this_epoch += loss_batch

            # TODO: Your code here
            # Mini-batch gradient descent
            w -= alpha*g - alpha*decay*w

        # TODO: Your code here
        # monitor model behavior after each epoch
        # 1. Compute the training loss by averaging loss_this_epoch
        training_loss = loss_this_epoch/int(np.ceil(N_train/batch_size))
        train_losses.append(training_loss)
        # 2. Perform validation on the validation test by the risk
        _, _, acc, _ = predict(X_val, w, t_val)
        valid_accs.append(acc)
        # 3. Keep track of the best validation epoch, risk, and the weights
        #print(risks_val)
        print('Epoch {} finished' .format(epoch))
        if acc_best <= acc:
            epoch_best = epoch
            acc_best = acc
            W_best = w

    # Return some variables as needed
    return epoch_best, acc_best, W_best, train_losses, valid_accs

def softmax(z):
    max_z = np.max(z, axis=1)
    max_z = max_z[:, np.newaxis]
    sum_z = np.sum(np.exp(z-max_z), axis=1)
    sum_z = sum_z[:, np.newaxis] 
    return np.exp(z-max_z) / sum_z
##############################
#Main code starts here
N_class = 10
alpha   = 0.1      # learning rate
batch_size   = 300    # batch size
MaxEpoch = 50        # Maximum epoch
decay = 0          # weight decay

def main():
    X_train, t_train, X_val, t_val, X_test, t_test = readMNISTdata()
    epoch_best, acc_best, W_best, losses_train, acc_train = train(X_train, t_train, X_val, t_val)
    _, _, acc_test, _ = predict(X_test, W_best, t_test)

    print('Best epoch', epoch_best)
    print('Test performance:', acc_test)
    print('Validation performance: ', acc_best)

    plt.subplots(figsize=(16, 4))
    plt.subplot(1,2,1)
    plt.plot(losses_train, label="Training Cross-Entropy Loss")
    plt.title("Loss")
    plt.xlabel('Number of epoch')
    plt.ylabel('Training Loss')

    plt.subplot(1,2,2)
    plt.plot(acc_train, label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')

    plt.savefig('Question 3' + '.jpg')

if __name__ == '__main__':
    main()