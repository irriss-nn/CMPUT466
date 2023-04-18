#!/usr/bin/env python3

import pickle as pickle
import numpy as np
import matplotlib.pyplot as plt

def predict(X, w, y=None):
    # X_new: Nsample x (d+1)
    # w: (d+1) x 1
    # y_new: Nsample
    y_hat = np.dot(X, w)
    # TODO: Your code here
    if y is not None:
        loss = 1/2 *(np.mean((y - y_hat)**2))

        #risk
        y_new = y*std_y + mean_y
        y_hat_norm = y_hat*std_y+mean_y
       
        abs_diff = np.absolute(y_hat_norm-y_new)
        risk = np.mean(abs_diff)

    else: 
        loss = None
        risk = None
    return y_hat, loss, risk


def train(X_train, y_train, X_val, y_val, decay):
    N_train = X_train.shape[0]
    N_val = X_val.shape[0]

    # initialization
    w = np.zeros([X_train.shape[1], 1])
    # w: (d+1)x1

    losses_train = []
    risks_val = []

    w_best = None
    risk_best = 10000
    epoch_best = 0

    for epoch in range(MaxIter):

        loss_this_epoch = 0
        for b in range(int(np.ceil(N_train/batch_size))):

            X_batch = X_train[b*batch_size: (b+1)*batch_size]
            y_batch = y_train[b*batch_size: (b+1)*batch_size]

            y_hat_batch, loss_batch, _ = predict(X_batch, w, y_batch)
            loss_this_epoch += loss_batch

            # TODO: Your code here
            # Mini-batch gradient descent
            gradient = (1/len(y_batch))*X_batch.T.dot(y_hat_batch-y_batch) + decay*w
            w = w - alpha*gradient


        # TODO: Your code here
        # monitor model behavior after each epoch
        # 1. Compute the training loss by averaging loss_this_epoch
        loss_train =  loss_this_epoch/batch_size
        losses_train.append(loss_train)
        # 2. Perform validation on the validation set by the risk
        y_hat, loss, risk_val = predict(X_val, w, y_val)
        risks_val.append(risk_val)
        # 3. Keep track of the best validation epoch, risk, and the weights
        if risk_val < risk_best:
            w_best = w
            risk_best = risk_val
            epoch_best = epoch
    # Return some variables as needed
    return w_best, risk_best, epoch_best, losses_train, risks_val


############################
# Main code starts here
############################
# Load data
with open("housing.pkl", "rb") as f:
    (X, y) = pickle.load(f)

# X: sample x dimension
# y: sample x 1

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)


# Augment feature
# for the extension first combine X and X^2
X = np.concatenate((X,X**2),axis=1)
# add first column 1
X_ = np.concatenate((np.ones([X.shape[0], 1]), X), axis=1)
# print(X_)
# X_: Nsample x (d+1)

# normalize features:
mean_y = np.mean(y)
std_y = np.std(y)

y = (y - np.mean(y)) / np.std(y)

# print(X.shape, y.shape) # It's always helpful to print the shape of a variable


# Randomly shuffle the data
np.random.seed(314)
np.random.shuffle(X_)
np.random.seed(314)
np.random.shuffle(y)

X_train = X_[:300]
y_train = y[:300]

X_val = X_[300:400]
y_val = y[300:400]

X_test = X_[400:]
y_test = y[400:]

#####################
# setting

alpha = 0.001      # learning rate
batch_size = 10    # batch size
MaxIter = 100        # Maximum iteration
decay = [3, 1, 0.3, 0.1, 0.03, 0.01] 
risks_val = []  
l = []      # weight decay

for i in decay:
    w_best, risk_best, epoch_best, losses_train, risks_val_dum = train(X_train, y_train, X_val, y_val, i)
    risks_val.append(risk_best)
    l.append(losses_train)

j = np.argmin(risks_val)
best_decay = decay[j]

print("Best hyperparameter: {}".format(best_decay))


# TODO: Your code here

# Perform test by the weights yielding the best validation performance

y_hat, loss, risk_test = predict(X_test, w_best, y_test)
# Report numbers and draw plots as required.
#numbers
print("Best validation performance: {}".format(epoch_best))
print("The validation performance (risk) in that epoch: {}".format(risk_best))
print("The test performance (risk) in that epoch: {}".format(risk_test))

#plot
plt.subplots(figsize=(16, 4))
plt.subplot(1,2,1)
plt.plot(range(MaxIter), losses_train)
plt.xlabel("the number of epochs")
plt.ylabel("training loss")

plt.subplot(1,2,2)
plt.plot(range(MaxIter), risks_val_dum)
plt.xlabel("the number of epochs")
plt.ylabel("validation risk")

plt.savefig('Question2b' + '.jpg')
