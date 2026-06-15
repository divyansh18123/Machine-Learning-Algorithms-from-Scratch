import numpy as np
import matplotlib.pyplot as plt
from numpy import random


AX1 = np.random.multivariate_normal([-3, -3], np.identity(2), size = 200)
AX2 = np.random.multivariate_normal([3, 3], np.identity(2), size = 200)

BX1 = np.random.multivariate_normal([-3, -3], 3 * np.identity(2), size = 200)
BX2 = np.random.multivariate_normal([3, 3], 3 * np.identity(2), size = 200)

index1 = np.array(random.choice(200, size = 60, replace = False))
AX1_val = AX1[index1]
AX1_train = AX1[np.setdiff1d(np.arange(200), index1)]

index2 = np.array(random.choice(200, size = 60, replace = False))
AX2_val = AX2[index2]
AX2_train = AX2[np.setdiff1d(np.arange(200), index2)]

index3 = np.array(random.choice(200, size = 60, replace = False))
BX1_val = BX1[index3]
BX1_train = BX1[np.setdiff1d(np.arange(200), index3)]

index4 = np.array(random.choice(200, size = 60, replace = False))
BX2_val = BX2[index4]
BX2_train = BX2[np.setdiff1d(np.arange(200), index4)]


w_a = np.array([0, 0])
b_a = 0
n = 0.01
miss_a = []
AX_train = np.vstack([AX1_train, AX2_train])
AY_train = np.hstack([-np.ones(140), np.ones(140)])


for i in range(300):
    miss = 0

    perm = np.random.permutation(280)
    AX_train = AX_train[perm]
    AY_train = AY_train[perm]

    for j in range(280):
        if AY_train[j] * (np.dot(w_a, AX_train[j]) + b_a) <= 0:
            w_a = w_a + n * AY_train[j] * AX_train[j]
            b_a = b_a + n * AY_train[j]
            miss += 1


    miss_a.append(miss)
    if miss == 0:
        print("Early stop for A at epoch", i+1)
        break






w_b = np.array([0, 0])
b_b = 0
n = 0.01
miss_b = []
BX_train = np.vstack([BX1_train, BX2_train])
BY_train = np.hstack([-np.ones(140), np.ones(140)])


for i in range(300):
    miss = 0

    perm = np.random.permutation(280)
    BX_train = BX_train[perm]
    BY_train = BY_train[perm]

    for j in range(280):
        if BY_train[j] * (np.dot(w_b, BX_train[j]) + b_b) <= 0:
            w_b = w_b + n * BY_train[j] * BX_train[j]
            b_b = b_b + n * BY_train[j]
            miss += 1


    miss_b.append(miss)
    if miss == 0:
        print("Early stop for B at epoch", i+1)
        break


plt.subplot(1,2,1)
plt.plot(range(1, len(miss_a) + 1, 1), miss_a)
plt.title("A")
plt.subplot(1,2,2)
plt.plot(range(1, len(miss_b) + 1, 1), miss_b)
plt.title("B")
plt.show()





AX_val = np.vstack([AX1_val, AX2_val])
AY_val = np.hstack([-np.ones(60), np.ones(60)])

plt.subplot(2,2,1)
plt.scatter(AX_train[:, 0], AX_train[:, 1], c = AY_train)
plt.title("A train")

x_line = np.linspace(-8, 8, 100)
y = -(w_a[0] * x_line + b_a) / (w_a[1] + 1e-10)
plt.plot(x_line, y)


plt.subplot(2,2,2)
plt.scatter(AX_val[:, 0], AX_val[:, 1], c = AY_val)
plt.title("A val")

x_line = np.linspace(-8, 8, 100)
y = -(w_a[0] * x_line + b_a) / (w_a[1] + 1e-10)
plt.plot(x_line, y)




BX_val = np.vstack([BX1_val, BX2_val])
BY_val = np.hstack([-np.ones(60), np.ones(60)])

plt.subplot(2,2,3)
plt.scatter(BX_train[:, 0], BX_train[:, 1], c = BY_train)
plt.title("B train")

x_line = np.linspace(-8, 8, 100)
y = -(w_b[0] * x_line + b_b) / (w_b[1] + 1e-10)
plt.plot(x_line, y)


plt.subplot(2,2,4)
plt.scatter(BX_val[:, 0], BX_val[:, 1], c = BY_val)
plt.title("B val")

x_line = np.linspace(-8, 8, 100)
y = -(w_b[0] * x_line + b_b) / (w_b[1] + 1e-10)
plt.plot(x_line, y)


plt.show()



correct_a = 0
for i in range(60):
    if np.dot(w_a, AX1_val[i]) + b_a <= 0:
        correct_a += 1
for i in range(60):
    if np.dot(w_a, AX2_val[i]) + b_a > 0:
        correct_a += 1
print("Test accuracy for A: ", correct_a/120)

correct_b = 0
for i in range(60):
    if np.dot(w_b, BX1_val[i]) + b_b <= 0:
        correct_b += 1
for i in range(60):
    if np.dot(w_b, BX2_val[i]) + b_b > 0:
        correct_b += 1
print("Test accuracy for B: ", correct_b/120)



