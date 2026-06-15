import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import random

with np.load("mnist.npz") as data:

    whole_x_test = data['x_test']
    whole_y_test = data['y_test']
    whole_x_train = data['x_train']
    whole_y_train = data['y_train']


x4_test = []
x4_train = []
x9_test = []
x9_train = []

for i in range(len(whole_y_test)):
    if whole_y_test[i] == 4:
        x4_test.append(whole_x_test[i])
    elif whole_y_test[i] == 9:
        x9_test.append(whole_x_test[i])

for i in range(len(whole_y_train)):
    if whole_y_train[i] == 4:
        x4_train.append(whole_x_train[i])
    elif whole_y_train[i] == 9:
        x9_train.append(whole_x_train[i])

x4_test = np.array(x4_test)
x4_train = np.array(x4_train)
x9_test = np.array(x9_test)
x9_train = np.array(x9_train)


x4_test = x4_test.reshape(len(x4_test), -1)
x4_train = x4_train.reshape(len(x4_train), -1)
x9_test = x9_test.reshape(len(x9_test), -1)
x9_train = x9_train.reshape(len(x9_train), -1)

x4_test = x4_test/255.0
x4_train = x4_train/255.0
x9_test = x9_test/255.0
x9_train = x9_train/255.0


def pca(X, p):
    u = np.zeros(784)
    for i in X:
        u += i
    u = u / len(X)

    Xc = np.zeros((len(X), 784))
    for i in range(len(X)):
        Xc[i] = X[i] - u

    S = np.matmul(Xc.T, Xc)/ (len(X)-1)
    lambd, U = np.linalg.eigh(S)

    idx = np.argsort(lambd)[::-1]
    lambd = lambd[idx]
    U = U[:, idx]


    U_p = U[:, :p]

    Y = np.matmul(Xc, U_p)

    return Y, U_p, u


index1 = np.array(random.choice(len(x4_train), size = 1000, replace = False))
x4_val = x4_train[index1]
x4_train = x4_train[np.setdiff1d(np.arange(len(x4_train)), index1)]

index2 = np.array(random.choice(len(x9_train), size = 1000, replace = False))
x9_val = x9_train[index2]
x9_train = x9_train[np.setdiff1d(np.arange(len(x9_train)), index2)]



X_train = np.vstack([x4_train, x9_train])
Y_train = np.hstack([-np.ones(len(x4_train)), np.ones(len(x9_train))])

X_val = np.vstack([x4_val, x9_val])
Y_val = np.hstack([-np.ones(len(x4_val)), np.ones(len(x9_val))])
X_test = np.vstack([x4_test, x9_test])
Y_test = np.hstack([-np.ones(len(x4_test)), np.ones(len(x9_test))])

X_train, U_5, u = pca(X_train, 5)


x_testc = np.zeros((len(X_test), 784))
for i in range(len(X_test)):
    x_testc[i] = X_test[i] - u
X_test = np.matmul(x_testc, U_5)

x_valc = np.zeros((len(X_val), 784))
for i in range(len(X_val)):
    x_valc[i] = X_val[i] - u
X_val = np.matmul(x_valc, U_5)




def single_stump(train, train_label, w, w_sum):
    ans_feature = 0
    ans_threshhold = (train[:, 0][0] + train[:, 0][1])/2
    min_error = float('inf')
    final_left = -1
    final_right = 1


    for i in range(5):
        feature = i

        index = np.argsort(train[:, feature])
        X = train[index]
        Y = train_label[index]
        W = w[index]

        index2 = np.array(random.choice(len(X) - 1, size = 1000, replace = False))
        for j in index2:
            cs = (X[j][feature] + X[j + 1][feature])/2
            
            left = X[:, feature] <= cs
            right = X[:, feature] > cs
            left_y = Y[left]
            right_y = Y[right]
            left_w = W[left]
            right_w = W[right]
            
            

            majority_left = np.sum(left_y)
            if majority_left >= 0:
                majority_left = 1
            else:
                majority_left = -1
            

            majority_right = np.sum(right_y)
            if majority_right >= 0:
                majority_right = 1
            else:
                majority_right = -1


            left_miss = left_y != majority_left
            error_left = np.sum(left_w * left_miss)

            right_miss = right_y != majority_right
            error_right = np.sum(right_w * right_miss)

            error = (error_left + error_right) / w_sum
            error = np.clip(error, 1e-10, 1 - 1e-10)


            if error <= min_error:
                min_error = error
                ans_feature = feature
                ans_threshhold = cs
                final_left = majority_left
                final_right = majority_right


    return ans_feature, ans_threshhold, min_error, final_left, final_right


w = [1/len(Y_train)] * len(Y_train)
w = np.array(w)
w_sum = 1
alpha_arr = []


ans_feature_arr = []
ans_threshhold_arr = []
min_error_arr = []
final_left_arr = []
final_right_arr = []


for t in range(300):
    ans_feature, ans_threshhold, min_error, final_left, final_right = single_stump(X_train, Y_train, w, w_sum)

    ans_feature_arr.append(ans_feature)
    ans_threshhold_arr.append(ans_threshhold)
    min_error_arr.append(min_error)
    final_left_arr.append(final_left)
    final_right_arr.append(final_right)




    alpha_arr.append(0.5 * math.log((1 - min_error)/max(min_error,  1e-10)))

    pred = []
    for i in range(len(X_train)):
        if X_train[i][ans_feature_arr[t]] <= ans_threshhold_arr[t]:
            pred.append(final_left_arr[t])
        else:
            pred.append(final_right_arr[t])

    w2 = []
    for b in range(len(pred)):
        if Y_train[b] == pred[b]:
            w2.append(w[b])
        else:
            w2.append(w[b] * np.exp(2 * alpha_arr[t]))
    w = np.array(w2)
    w_sum = np.sum(w)
    
    



val_acc_arr = []
for t in range(300):
    correct = 0
    for i in range(len(X_val)):
        prediction = 0
        for q in range(t + 1):
            if q == 300:
                continue
            if X_val[i][ans_feature_arr[q]] <= ans_threshhold_arr[q]:
                prediction += alpha_arr[q] * final_left_arr[q]
            else:
                prediction += alpha_arr[q] * final_right_arr[q]
        if prediction >= 0: 
            final_pred = 1
        else:
            final_pred = -1
        
        if Y_val[i] == final_pred:
            correct += 1

    val_acc_arr.append(correct/len(Y_val))

maxmum = 0
maxindex = 0
for i in range(len(val_acc_arr)):
    if val_acc_arr[i] > maxmum:
        maxmum = val_acc_arr[i]
        maxindex = i


correct = 0
for i in range(len(X_test)):
    prediction = 0
    for q in range(maxindex + 1):
        if q == 300:
            continue

        if X_test[i][ans_feature_arr[q]] <= ans_threshhold_arr[q]:
            prediction += alpha_arr[q] * final_left_arr[q]
        else:
            prediction += alpha_arr[q] * final_right_arr[q]
    if prediction >= 0: 
        final_pred = 1
    else:
        final_pred = -1
    
    if Y_test[i] == final_pred:
        correct += 1
test_acc = correct / len(X_test)
print("The maximum val accuracy iteration is ", maxindex + 1)
print("with accuracy ", maxmum)
print("Test accuracy on this is ", test_acc)



plt.plot(val_acc_arr)
plt.title("Validation Accuracy")
plt.show()