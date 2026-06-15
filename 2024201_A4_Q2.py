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


def single_stump(train, train_label):
    ans_feature = None
    ans_threshhold = None
    min_ssr = float('inf')
    final_left_avg = 0
    final_right_avg = 0


    for i in range(5):
        feature = i

        index = np.argsort(train[:, feature])
        X = train[index]
        Y = train_label[index]

        index2 = np.array(random.choice(len(X) - 1, size = 1000, replace = False))
        for j in index2:
            cs = (X[j][feature] + X[j + 1][feature])/2
            
            left = X[:, feature] <= cs
            right = X[:, feature] > cs
            
            if np.sum(left) == 0 or np.sum(right) == 0:
                continue
            left_avg = np.mean(Y[left])
            right_avg = np.mean(Y[right])


            SSR = np.sum((Y[left] - left_avg)**2) + np.sum((Y[right] - right_avg)**2)

            if SSR <= min_ssr:
                min_ssr = SSR
                ans_feature = feature
                ans_threshhold = cs
                final_left_avg = left_avg
                final_right_avg = right_avg

    return ans_feature, ans_threshhold, min_ssr, final_left_avg, final_right_avg


def prediction_h(sifeature, sithreshhold, sileft_avg, siright_avg, data_point):
    if data_point[int (sifeature)] <= sithreshhold:
        return sileft_avg
    else:
        return siright_avg


def main_func(n, X_train, Y_train, X_val, Y_val, X_test, Y_test):

    Y_train2 = Y_train.copy()
    ans_feature_arr = []
    ans_threshhold_arr = []
    min_ssr_arr = []
    final_left_arr = []
    final_right_arr = []




    for t in range(300):

        ans_feature, ans_threshhold, min_ssr, final_left_avg, final_right_avg = single_stump(X_train, Y_train2)

        ans_feature_arr.append(ans_feature)
        ans_threshhold_arr.append(ans_threshhold)
        min_ssr_arr.append(min_ssr)
        final_left_arr.append(final_left_avg)
        final_right_arr.append(final_right_avg)

        for i in range(len(X_train)):
            Y_train2[i] = Y_train2[i] - n * prediction_h(ans_feature_arr[t], ans_threshhold_arr[t], final_left_arr[t], final_right_arr[t], X_train[i])





    val_mse_arr = []
    for t in range(300):
        mse_err = 0
        for i in range(len(X_val)):
            prediction = 0
            for q in range(t+1):
                prediction += n * prediction_h(ans_feature_arr[q], ans_threshhold_arr[q], final_left_arr[q], final_right_arr[q], X_val[i])
            
            mse_err += (prediction - Y_val[i])**2

        val_mse_arr.append(mse_err/len(Y_val))


    minmum = float('inf')
    minindex = 0
    for i in range(len(val_mse_arr)):
        if val_mse_arr[i] < minmum:
            minmum = val_mse_arr[i]
            minindex = i




    test_mse_err = 0
    for i in range(len(X_test)):
        prediction = 0
        for q in range(minindex + 1):
            prediction += n * prediction_h(ans_feature_arr[q], ans_threshhold_arr[q], final_left_arr[q], final_right_arr[q], X_test[i])
        
        test_mse_err += (prediction - Y_test[i])**2

    test_mse_err = test_mse_err/ len(X_test)


    train_mse_err = 0
    for i in range(len(X_train)):
        prediction = 0
        for q in range(minindex + 1):
            prediction += n * prediction_h(ans_feature_arr[q], ans_threshhold_arr[q], final_left_arr[q], final_right_arr[q], X_train[i])
        
        train_mse_err += (prediction - Y_train[i])**2

    train_mse_err = train_mse_err/ len(X_train)

    print("n is ", n)
    print("The minimum val mse iteration is ", minindex + 1)
    print("with mse ", minmum)
    print("Test mse on this is ", test_mse_err)
    print("Train mse on this is ", train_mse_err)


    plt.plot(val_mse_arr)
    plt.title(f"Validation MSE {n}")
    plt.show()


main_func(0.001, X_train, Y_train, X_val, Y_val, X_test, Y_test)
main_func(0.01, X_train, Y_train, X_val, Y_val, X_test, Y_test)
main_func(0.1, X_train, Y_train, X_val, Y_val, X_test, Y_test)
main_func(0.2, X_train, Y_train, X_val, Y_val, X_test, Y_test)
main_func(0.5, X_train, Y_train, X_val, Y_val, X_test, Y_test)
main_func(1, X_train, Y_train, X_val, Y_val, X_test, Y_test)
